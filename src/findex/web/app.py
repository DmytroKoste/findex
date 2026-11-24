import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.store import load_index
from findex.web.config import Settings
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
)
from findex.web.service import (
    highlight_snippet_html,
    load_doc_text,
    run_paginated_search,
    total_pages,
)

log = logging.getLogger(__name__)
_WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
templates.env.filters["highlight"] = highlight_snippet_html


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = Settings()
        request.app.state.settings = settings
    return cast(Settings, settings)


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "index", None)
    if index is None or getattr(index, "is_closed", False):
        raise HTTPException(status_code=503, detail="Індекс ще не завантажений")
    return cast(Index, index)


IndexDep = Annotated[Index, Depends(get_index)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = getattr(app.state, "settings", None)
    if settings is None:
        settings = Settings()
        app.state.settings = settings
    app.state.started_at = time.monotonic()

    if getattr(app.state, "index", None) is None:
        idx_path = Path(settings.index_path)
        log.info("Завантаження індексу з %s", idx_path)
        app.state.index = load_index(idx_path)

    try:
        yield
    finally:
        idx = getattr(app.state, "index", None)
        if idx is not None and hasattr(idx, "close"):
            idx.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings()
    application = FastAPI(
        title="findex Search Engine",
        version="0.7.0",
        description="HTTP API та веб-інтерфейс пошукового рушія findex (BM25 / TF-IDF)",
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    application.state.started_at = time.monotonic()

    origins = [o.strip() for o in active_settings.cors_origins.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    application.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "Неочікувана помилка обробки запиту [request_id=%s] %s %s",
                req_id,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Внутрішня помилка сервера", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = req_id
        log.info(
            "[request_id=%s] %s %s -> %d (%.2f ms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        index = getattr(request.app.state, "index", None)
        if index is None or getattr(index, "is_closed", False):
            raise HTTPException(status_code=503, detail="Індекс не завантажений")
        return HealthResponse(status="ok")

    @application.get("/stats", response_model=StatsResponse, tags=["system"])
    def stats(request: Request, index: IndexDep) -> StatsResponse:
        started_at = float(getattr(request.app.state, "started_at", time.monotonic()))
        uptime = max(0.0, time.monotonic() - started_at)
        return StatsResponse(
            num_docs=index.num_docs,
            vocab_size=len(index),
            avg_doc_length=round(index.avg_doc_length, 2),
            uptime_s=round(uptime, 2),
        )

    @application.get("/search", response_model=SearchResponse, tags=["search"])
    def search_api(
        index: IndexDep,
        q: str = Query(..., min_length=1, max_length=200, description="Пошуковий запит (AND, OR, NOT, фрази)"),
        k: int = Query(10, ge=1, le=100, description="Кількість результатів на сторінку"),
        scorer: Literal["bm25", "tfidf"] = Query("bm25", description="Функція ранжування"),
        page: int = Query(1, ge=1, le=1000, description="Номер сторінки"),
    ) -> SearchResponse:
        t0 = time.perf_counter()
        try:
            total, raw_items = run_paginated_search(index, q, scorer, k, page)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Некоректний синтаксис запиту: {exc}") from exc

        took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        items = [
            SearchResultItem(doc_id=doc_id, title=title, score=score, snippet=snippet)
            for doc_id, title, score, snippet in raw_items
        ]
        return SearchResponse(
            query=q,
            scorer=scorer,
            page=page,
            k=k,
            total=total,
            took_ms=took_ms,
            results=items,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["documents"])
    def get_document_api(doc_id: int, index: IndexDep) -> DocumentResponse:
        if doc_id not in index.doc_meta:
            raise HTTPException(status_code=404, detail=f"Документ doc_id={doc_id} не знайдено")
        meta = index.doc_meta[doc_id]
        content = load_doc_text(index, doc_id)
        return DocumentResponse(
            doc_id=doc_id,
            title=meta.title,
            path=meta.path,
            length=index.doc_length(doc_id),
            content=content,
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home_ui(
        request: Request,
        q: str = Query("", max_length=200),
        k: int = Query(10, ge=1, le=50),
        scorer: Literal["bm25", "tfidf"] = Query("bm25"),
        page: int = Query(1, ge=1, le=1000),
    ) -> HTMLResponse:
        index = getattr(request.app.state, "index", None)
        cleaned_q = q.strip()
        results: list[SearchResultItem] = []
        total = 0
        took_ms = 0.0
        error_msg: str | None = None

        if index is None or getattr(index, "is_closed", False):
            error_msg = "Індекс ще не завантажено на сервері."
        elif cleaned_q:
            t0 = time.perf_counter()
            try:
                total, raw_items = run_paginated_search(index, cleaned_q, scorer, k, page)
                took_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                results = [
                    SearchResultItem(doc_id=doc_id, title=title, score=score, snippet=snippet)
                    for doc_id, title, score, snippet in raw_items
                ]
            except ValueError as exc:
                error_msg = f"Помилка синтаксису запиту: {exc}"

        pages_count = total_pages(total, k)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": cleaned_q,
                "k": k,
                "scorer": scorer,
                "page": page,
                "total": total,
                "pages_count": pages_count,
                "took_ms": took_ms,
                "results": results,
                "error_msg": error_msg,
                "num_docs": index.num_docs if index is not None else 0,
                "vocab_size": len(index) if index is not None else 0,
            },
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_ui(request: Request, doc_id: int, index: IndexDep) -> HTMLResponse:
        if doc_id not in index.doc_meta:
            raise HTTPException(status_code=404, detail=f"Документ doc_id={doc_id} не знайдено")
        meta = index.doc_meta[doc_id]
        content = load_doc_text(index, doc_id)
        return templates.TemplateResponse(
            request=request,
            name="doc.html",
            context={
                "doc_id": doc_id,
                "title": meta.title,
                "path": meta.path,
                "length": index.doc_length(doc_id),
                "content": content,
            },
        )

    return application


app = create_app()
