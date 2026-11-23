from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from findex.corpus import iter_documents
from findex.crawler import (
    DEFAULT_USER_AGENT,
    Page,
    RobotsCache,
    canonicalize_url,
    crawl,
    fetch,
)
from findex.index import build_index


def test_canonicalize_url() -> None:
    assert (
        canonicalize_url("HTTP://Example.COM:80/docs/page/?b=2&a=1#section") == "http://example.com/docs/page?a=1&b=2"
    )
    assert canonicalize_url("/sub/", base_url="https://example.com/root") == "https://example.com/sub"
    assert canonicalize_url("mailto:test@example.com") is None


def test_fetch_retries_429_and_503_but_not_404() -> None:
    async def _run() -> None:
        calls: dict[str, int] = {"/rate": 0, "/flaky": 0, "/missing": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("User-Agent") == DEFAULT_USER_AGENT
            path = request.url.path
            calls[path] += 1
            if path == "/rate":
                if calls[path] == 1:
                    return httpx.Response(429, headers={"Retry-After": "0.01"}, text="slow down")
                return httpx.Response(200, text="recovered 429")
            if path == "/flaky":
                if calls[path] < 3:
                    return httpx.Response(503, text="unavailable")
                return httpx.Response(200, text="recovered 503")
            return httpx.Response(404, text="not found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            r429 = await fetch(client, "https://example.com/rate", base_delay=0.01, max_delay=0.02)
            assert r429.status == 200
            assert r429.attempts == 2
            assert r429.text == "recovered 429"

            r503 = await fetch(client, "https://example.com/flaky", base_delay=0.01, max_delay=0.02)
            assert r503.status == 200
            assert r503.attempts == 3
            assert r503.text == "recovered 503"

            r404 = await fetch(client, "https://example.com/missing", base_delay=0.01, max_delay=0.02)
            assert r404.status == 404
            assert r404.attempts == 1
            assert calls["/missing"] == 1

    asyncio.run(_run())


def test_fetch_handles_timeout() -> None:
    async def _run() -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("simulated read timeout", request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/slow",
                timeout=0.05,
                max_retries=2,
                base_delay=0.01,
                max_delay=0.02,
            )
            assert res.status == 0
            assert res.attempts == 2
            assert res.error is not None and "Timeout" in res.error
            assert attempts == 2

    asyncio.run(_run())


def test_robots_cache_fetches_once_and_enforces_rules() -> None:
    async def _run() -> None:
        robots_requests = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal robots_requests
            if request.url.path == "/robots.txt":
                robots_requests += 1
                return httpx.Response(
                    200,
                    text="User-agent: *\nDisallow: /private/\nAllow: /\n",
                )
            return httpx.Response(200, text="ok")

        cache = RobotsCache()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            assert await cache.allowed(client, "https://docs.example.com/public/page1") is True
            assert await cache.allowed(client, "https://docs.example.com/private/secret") is False
            assert await cache.allowed(client, "https://docs.example.com/public/page2") is True
            assert robots_requests == 1

    asyncio.run(_run())


def test_crawl_respects_max_pages_robots_dedup_and_feeds_index(tmp_path: Path) -> None:
    async def _run() -> None:
        requested_urls: list[str] = []

        pages_html: dict[str, str] = {
            "/": """
                <html><head><title>Home</title></head>
                <body>
                    <h1>Async Crawler Home</h1>
                    <a href="/a">Page A</a>
                    <a href="/a#dup">Page A duplicate</a>
                    <a href="/b">Page B</a>
                    <a href="/blocked">Blocked</a>
                    <a href="https://external.org/out">External</a>
                </body></html>
            """,
            "/a": """
                <html><head><title>Page A</title></head>
                <body><p>Coroutines and event loop</p><a href="/c">Page C</a></body></html>
            """,
            "/b": """
                <html><head><title>Page B</title></head>
                <body><p>Structured concurrency with TaskGroup</p><a href="/d">Page D</a></body></html>
            """,
            "/c": "<html><head><title>Page C</title></head><body><p>Semaphores</p></body></html>",
            "/d": "<html><head><title>Page D</title></head><body><p>Queues</p></body></html>",
            "/blocked": "<html><body>Should never be fetched</body></html>",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
            requested_urls.append(str(request.url))
            html = pages_html.get(path)
            if html is None:
                return httpx.Response(404, text="missing")
            return httpx.Response(200, text=html)

        log_path = tmp_path / "crawl.log"
        jsonl_path = tmp_path / "crawl.jsonl"
        transport = httpx.MockTransport(handler)

        collected: list[Page] = []
        with jsonl_path.open("w", encoding="utf-8") as fp:
            async for page in crawl(
                ["https://example.com/"],
                max_pages=3,
                concurrency=3,
                per_host=2,
                per_host_delay=0.0,
                allowed_domains={"example.com"},
                log_path=log_path,
                transport=transport,
            ):
                collected.append(page)
                fp.write(
                    json.dumps(
                        {
                            "url": page.url,
                            "title": page.title,
                            "text": page.text,
                            "fetched_at": page.fetched_at,
                        }
                    )
                    + "\n"
                )

        assert len(collected) == 3
        urls = [p.url for p in collected]
        assert len(urls) == len(set(urls))
        assert "https://example.com/blocked" not in requested_urls
        assert all("external.org" not in u for u in requested_urls)
        assert log_path.exists()

        docs = list(iter_documents(jsonl_path))
        assert len(docs) == 3

        ix = build_index(jsonl_path, workers=2, executor="threads")
        assert ix.num_docs == 3
        assert "crawler" in ix or "coroutines" in ix or "taskgroup" in ix

    asyncio.run(_run())
