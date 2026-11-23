from __future__ import annotations

import asyncio
import json
import tracemalloc
from pathlib import Path

from findex.crawler import CrawlStats, crawl

SEED = "https://docs.python.org/3/tutorial/"
MAX_PAGES = 200


async def run_one(concurrency: int) -> tuple[int, float, float, int, float]:
    tracemalloc.start()
    stats = CrawlStats()
    out_path = Path(f"data/bench_c{concurrency}.jsonl")
    log_path = Path(f"data/bench_c{concurrency}.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_host = 1 if concurrency == 1 else min(5, concurrency)
    with out_path.open("w", encoding="utf-8") as fp:
        async for page in crawl(
            [SEED],
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host=per_host,
            per_host_delay=0.05,
            log_path=log_path,
            stats=stats,
        ):
            fp.write(
                json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    return stats.fetched, stats.elapsed, stats.pages_per_sec, stats.errors, peak_mb


async def main() -> None:
    print("| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS (MB) |")
    print("|---|---|---|---|---|---|")
    for c in (1, 5, 20):
        pages, wall, pps, errors, peak_mb = await run_one(c)
        print(f"| {c} | {pages} | {wall:.2f} | {pps:.2f} | {errors} | {peak_mb:.2f} |")


if __name__ == "__main__":
    asyncio.run(main())
