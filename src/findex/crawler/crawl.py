from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import DEFAULT_USER_AGENT, RobotsCache


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    started_at: float = field(default_factory=time.perf_counter)
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    last_error: str | None = None

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_sec(self) -> float:
        return self.fetched / self.elapsed


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Normalize URL: lowercase scheme/host, strip fragment, sort query, normalize trailing slash."""
    joined = urljoin(base_url, raw_url) if base_url else raw_url
    parts = urlsplit(joined)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return None

    netloc = parts.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, sorted_query, ""))


def extract_page_and_links(html: str, page_url: str) -> tuple[str, str, list[str]]:
    """Extract title, clean visible text, and canonical outgoing links using selectolax lexbor."""
    tree = LexborHTMLParser(html)

    title = page_url
    title_node = tree.css_first("title")
    if title_node is not None:
        raw_title = title_node.text(strip=True)
        if raw_title:
            title = raw_title[:200]

    links: list[str] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if not href:
            continue
        canon = canonicalize_url(href, base_url=page_url)
        if canon is not None:
            links.append(canon)

    for bad_tag in tree.css("script, style, noscript, svg"):
        bad_tag.decompose()

    body = tree.body or tree.root
    raw_text = body.text(separator=" ", strip=True) if body is not None else ""
    clean_text = " ".join(raw_text.split())
    return title, clean_text, links


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.2,
    allowed_domains: set[str] | None = None,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_USER_AGENT,
    log_path: Path | None = None,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Page]:
    """Polite async web crawler yielding Page objects as they arrive."""
    if max_pages < 1:
        return

    crawl_stats = stats if stats is not None else CrawlStats()
    crawl_stats.started_at = time.perf_counter()

    frontier: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=max(1, concurrency * 2))
    seen: set[str] = set()
    domains: set[str] = set(d.lower() for d in allowed_domains) if allowed_domains else set()

    for seed in seeds:
        canon = canonicalize_url(seed)
        if canon is not None and canon not in seen:
            seen.add(canon)
            frontier.put_nowait(canon)
            if not allowed_domains:
                domains.add(urlsplit(canon).netloc)

    crawl_stats.queue_depth = frontier.qsize()
    if frontier.empty():
        return

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    host_next_allowed: dict[str, float] = {}
    robots = RobotsCache(user_agent=user_agent, timeout=timeout)

    state_lock = asyncio.Lock()
    enqueued_pages = 0
    stop_event = asyncio.Event()

    log_fp = log_path.open("w", encoding="utf-8") if log_path is not None else None

    def _write_log(res: FetchResult, note: str = "") -> None:
        if log_fp is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "url": res.url,
            "status": res.status,
            "bytes": res.num_bytes,
            "elapsed_ms": round(res.elapsed * 1000.0, 2),
            "attempts": res.attempts,
            "error": res.error or note or None,
        }
        log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_fp.flush()

    def _is_domain_allowed(url: str) -> bool:
        if not domains:
            return True
        host = urlsplit(url).netloc.lower()
        return any(host == d or host.endswith(f".{d}") for d in domains)

    async def _wait_host_turn(host: str) -> None:
        if per_host_delay <= 0.0:
            return
        lock = host_rate_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.perf_counter()
            ready_at = host_next_allowed.get(host, now)
            if ready_at > now:
                await asyncio.sleep(ready_at - now)
            host_next_allowed[host] = time.perf_counter() + per_host_delay

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal enqueued_pages
        while not stop_event.is_set():
            try:
                url = await asyncio.wait_for(frontier.get(), timeout=0.1)
            except TimeoutError:
                if frontier.empty() and crawl_stats.in_flight == 0:
                    return
                continue

            crawl_stats.in_flight += 1
            crawl_stats.queue_depth = frontier.qsize()
            try:
                if stop_event.is_set() or not _is_domain_allowed(url):
                    continue

                if not await robots.allowed(client, url):
                    _write_log(
                        FetchResult(url, url, 0, "", 0, 0.0, 0, "Disallowed by robots.txt"),
                        note="robots_disallow",
                    )
                    continue

                host = urlsplit(url).netloc.lower()
                host_sem = host_sems.setdefault(host, asyncio.Semaphore(max(1, per_host)))

                async with global_sem, host_sem:
                    if stop_event.is_set():
                        continue
                    await _wait_host_turn(host)
                    if stop_event.is_set():
                        continue

                    res = await fetch(
                        client,
                        url,
                        timeout=timeout,
                        user_agent=user_agent,
                    )

                _write_log(res)

                if res.error is not None or res.status >= 400:
                    crawl_stats.errors += 1
                    crawl_stats.last_error = f"{res.url} -> {res.error}"
                    continue

                final_canon = canonicalize_url(res.final_url) or url
                seen.add(final_canon)
                title, text, links = extract_page_and_links(res.text, final_canon)

                should_emit = False
                async with state_lock:
                    if enqueued_pages < max_pages:
                        enqueued_pages += 1
                        crawl_stats.fetched = enqueued_pages
                        should_emit = True
                        if enqueued_pages >= max_pages:
                            stop_event.set()

                if should_emit:
                    page = Page(
                        url=final_canon,
                        title=title,
                        text=text,
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                    await out_queue.put(page)

                if not stop_event.is_set():
                    for link in links:
                        if link not in seen and _is_domain_allowed(link):
                            seen.add(link)
                            frontier.put_nowait(link)
                    crawl_stats.queue_depth = frontier.qsize()
            finally:
                crawl_stats.in_flight -= 1
                frontier.task_done()
                crawl_stats.queue_depth = frontier.qsize()

    async def _run_pool() -> None:
        limits = httpx.Limits(
            max_connections=max(1, concurrency),
            max_keepalive_connections=max(1, concurrency),
        )
        try:
            async with httpx.AsyncClient(limits=limits, transport=transport) as client:
                try:
                    async with asyncio.TaskGroup() as tg:
                        workers = [tg.create_task(_worker(client)) for _ in range(max(1, concurrency))]
                        while not stop_event.is_set():
                            if frontier.empty() and crawl_stats.in_flight == 0:
                                stop_event.set()
                                break
                            await asyncio.sleep(0.05)
                        for w in workers:
                            w.cancel()
                except* asyncio.CancelledError:
                    pass
        finally:
            await out_queue.put(None)
            if log_fp is not None:
                log_fp.close()

    pool_task = asyncio.create_task(_run_pool())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not pool_task.done():
            pool_task.cancel()
        await asyncio.gather(pool_task, return_exceptions=True)
