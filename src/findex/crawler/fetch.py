from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx

from findex.crawler.robots import DEFAULT_USER_AGENT


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    num_bytes: int
    elapsed: float
    attempts: int
    error: str | None = None


def _parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    try:
        delay = float(header_value.strip())
        return max(0.0, delay)
    except ValueError:
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """Fetch a single URL politely with timeout, bounded retries, and exponential backoff."""
    t0 = time.perf_counter()
    last_status = 0
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(
                    url,
                    headers={"User-Agent": user_agent},
                    follow_redirects=True,
                )
            last_status = resp.status_code
            raw_bytes = resp.content

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_error = f"HTTP {resp.status_code}"
                if attempt < max_retries:
                    retry_after = (
                        _parse_retry_after(resp.headers.get("Retry-After")) if resp.status_code == 429 else None
                    )
                    backoff = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, 0.5)
                    await asyncio.sleep(retry_after if retry_after is not None else backoff)
                    continue

            return FetchResult(
                url=url,
                final_url=str(resp.url),
                status=resp.status_code,
                text=resp.text if resp.status_code < 400 else "",
                num_bytes=len(raw_bytes),
                elapsed=time.perf_counter() - t0,
                attempts=attempt,
                error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
            )

        except (TimeoutError, httpx.TimeoutException) as exc:
            last_status = 0
            last_error = f"Timeout: {type(exc).__name__}"
        except httpx.TooManyRedirects:
            return FetchResult(
                url=url,
                final_url=url,
                status=0,
                text="",
                num_bytes=0,
                elapsed=time.perf_counter() - t0,
                attempts=attempt,
                error="TooManyRedirects",
            )
        except httpx.HTTPError as exc:
            last_status = 0
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            backoff = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, 0.5)
            await asyncio.sleep(backoff)

    return FetchResult(
        url=url,
        final_url=url,
        status=last_status,
        text="",
        num_bytes=0,
        elapsed=time.perf_counter() - t0,
        attempts=max_retries,
        error=last_error or "Unknown fetch failure",
    )
