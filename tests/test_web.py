import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.web.app import create_app, get_index
from findex.web.config import Settings


def _make_fixture_index(tmp_path: Path) -> Index:
    p0 = tmp_path / "doc0.txt"
    p1 = tmp_path / "doc1.txt"
    p2 = tmp_path / "doc2.txt"
    p0.write_text("Python asyncio coroutines and event loops power fast web servers", encoding="utf-8")
    p1.write_text("FastAPI uses Pydantic type hints for validation and OpenAPI docs", encoding="utf-8")
    p2.write_text("BM25 ranking scores relevant python search documents accurately", encoding="utf-8")

    postings = {
        "python": [Posting(doc_id=0, tf=1), Posting(doc_id=2, tf=1)],
        "asyncio": [Posting(doc_id=0, tf=1)],
        "fastapi": [Posting(doc_id=1, tf=1)],
        "pydantic": [Posting(doc_id=1, tf=1)],
        "bm25": [Posting(doc_id=2, tf=1)],
        "search": [Posting(doc_id=2, tf=1)],
    }
    doc_lengths = {0: 10, 1: 9, 2: 8}
    doc_meta = {
        0: DocMeta(doc_id=0, path=str(p0), title="Asyncio Guide"),
        1: DocMeta(doc_id=1, path=str(p1), title="FastAPI Intro"),
        2: DocMeta(doc_id=2, path=str(p2), title="BM25 Ranking"),
    }
    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


def test_health_503_when_unloaded() -> None:
    app = create_app(Settings(index_path="nonexistent_index_file.idx"))
    client = cast(Any, TestClient(app))
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "detail" in resp.json()


def test_search_pagination_and_validation(tmp_path: Path) -> None:
    app = create_app()
    fixture_idx = _make_fixture_index(tmp_path)
    app.dependency_overrides[get_index] = lambda: fixture_idx

    with TestClient(app) as raw_client:
        client = cast(Any, raw_client)
        ok = client.get("/search", params={"q": "python", "k": 1, "page": 2, "scorer": "bm25"})
        assert ok.status_code == 200
        body = ok.json()
        assert body["total"] == 2
        assert body["page"] == 2
        assert len(body["results"]) == 1
        assert "X-Request-ID" in ok.headers

        bad_k = client.get("/search", params={"q": "python", "k": "abc"})
        assert bad_k.status_code == 422

        bad_scorer = client.get("/search", params={"q": "python", "scorer": "unknown"})
        assert bad_scorer.status_code == 422


def test_document_and_stats_and_ui(tmp_path: Path) -> None:
    app = create_app()
    fixture_idx = _make_fixture_index(tmp_path)
    app.state.index = fixture_idx
    app.dependency_overrides[get_index] = lambda: fixture_idx

    with TestClient(app) as raw_client:
        client = cast(Any, raw_client)
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        st = client.get("/stats")
        assert st.status_code == 200
        assert st.json()["num_docs"] == 3

        doc = client.get("/docs/1")
        assert doc.status_code == 200
        assert doc.json()["title"] == "FastAPI Intro"

        missing = client.get("/docs/999")
        assert missing.status_code == 404

        ui = client.get("/", params={"q": "fastapi"})
        assert ui.status_code == 200
        assert "FastAPI Intro" in ui.text
        assert "<mark>FastAPI</mark>" in ui.text


def test_sync_vs_async_concurrency_behavior() -> None:
    """Перевіряє, що блокуюча робота в async def серіалізує запити, а def виконується у thread pool."""
    demo = FastAPI()

    @demo.get("/blocking-async")
    async def blocking_async() -> dict[str, bool]:
        time.sleep(0.05)
        return {"ok": True}

    @demo.get("/blocking-sync")
    def blocking_sync() -> dict[str, bool]:
        time.sleep(0.05)
        return {"ok": True}

    async def _run_sync_parallel() -> float:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool, TestClient(demo) as raw_client:
            client = cast(Any, raw_client)
            futs = [pool.submit(client.get, "/blocking-sync") for _ in range(8)]
            for f in futs:
                assert f.result().status_code == 200
        return time.perf_counter() - t0

    async def _run_async_serial() -> float:
        transport = httpx.ASGITransport(app=demo)
        t0 = time.perf_counter()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resps = await asyncio.gather(*[client.get("/blocking-async") for _ in range(8)])
            assert all(r.status_code == 200 for r in resps)
        return time.perf_counter() - t0

    sync_elapsed = asyncio.run(_run_sync_parallel())
    async_elapsed = asyncio.run(_run_async_serial())
    assert sync_elapsed < async_elapsed
