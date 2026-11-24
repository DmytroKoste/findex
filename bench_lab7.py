import asyncio
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, math.ceil((pct / 100.0) * len(sorted_vals)) - 1)
    return sorted_vals[idx]


async def _wait_ready(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    for _ in range(8):
                        await client.get("/search", params={"q": "python", "k": 5})
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError(f"Server at {base_url} did not become ready")


async def _load_test(
    base_url: str,
    concurrency: int = 50,
    duration_s: float = 5.0,
) -> tuple[float, float, float, float]:
    latencies_ms: list[float] = []
    stop_at = time.perf_counter() + duration_s
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)

    async def worker(client: httpx.AsyncClient) -> None:
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            r = await client.get("/search", params={"q": "python", "k": 10})
            if r.status_code == 200:
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    t_start = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, limits=limits) as client:
        await asyncio.gather(*[worker(client) for _ in range(concurrency)])
    elapsed = time.perf_counter() - t_start

    latencies_ms.sort()
    rps = len(latencies_ms) / elapsed if elapsed > 0 else 0.0
    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = _percentile(latencies_ms, 95.0)
    p99 = _percentile(latencies_ms, 99.0)
    return rps, p50, p95, p99


def _run_server_case(
    label: str,
    module_target: str,
    port: int,
    workers: int,
    concurrency: int = 50,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        module_target,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "error",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    try:
        asyncio.run(_wait_ready(base_url))
        rps, p50, p95, p99 = asyncio.run(_load_test(base_url, concurrency=concurrency, duration_s=5.0))
        print(f"| {label} | `/search` | {concurrency} | {rps:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} |")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        deployed_url = sys.argv[1].rstrip("/")
        asyncio.run(_wait_ready(deployed_url, timeout=60.0))
        rps, p50, p95, p99 = asyncio.run(_load_test(deployed_url, concurrency=20, duration_s=10.0))
        print(f"| deployed | `/search` | 20 | {rps:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} |")
        return

    shim = Path("_bench_async_app.py")
    shim.write_text(
        """import time
from typing import Literal
from fastapi import Query
from findex.web.app import app
from findex.web.schemas import SearchResponse, SearchResultItem
from findex.web.service import run_paginated_search

app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/search"]


@app.get("/search", response_model=SearchResponse)
async def search_async_blocking(
    q: str = Query(..., min_length=1, max_length=200),
    k: int = Query(10, ge=1, le=100),
    scorer: Literal["bm25", "tfidf"] = Query("bm25"),
    page: int = Query(1, ge=1, le=1000),
) -> SearchResponse:
    index = app.state.index
    t0 = time.perf_counter()
    total, raw_items = run_paginated_search(index, q, scorer, k, page)
    took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        k=k,
        total=total,
        took_ms=took_ms,
        results=[SearchResultItem(doc_id=d, title=t, score=s, snippet=sn) for d, t, s, sn in raw_items],
    )
""",
        encoding="utf-8",
    )
    try:
        print("| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 |")
        print("|---|---|---|---|---|---|---|")
        _run_server_case("local, 1 worker, `async def`", "_bench_async_app:app", 8101, workers=1, concurrency=50)
        _run_server_case("local, 1 worker, `def`", "findex.web.app:app", 8102, workers=1, concurrency=50)
        _run_server_case("local, 4 workers", "findex.web.app:app", 8103, workers=4, concurrency=50)
    finally:
        shim.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
