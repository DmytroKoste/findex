from __future__ import annotations

import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

DEFAULT_USER_AGENT = "findex-bot/0.6 (+https://github.com/DmytroKoste/findex)"


class RobotsCache:
    """Fetches, caches, and evaluates robots.txt rules per host."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return False

        origin = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
        lock = self._locks.setdefault(origin, asyncio.Lock())

        async with lock:
            if origin not in self._parsers:
                self._parsers[origin] = await self._fetch_parser(client, origin)

        parser = self._parsers[origin]
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def _fetch_parser(self, client: httpx.AsyncClient, origin: str) -> RobotFileParser | None:
        robots_url = f"{origin}/robots.txt"
        try:
            async with asyncio.timeout(self.timeout):
                resp = await client.get(
                    robots_url,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True,
                )
        except (httpx.HTTPError, TimeoutError):
            return None

        if resp.status_code in (401, 403):
            rp = RobotFileParser()
            rp.parse(["User-agent: *", "Disallow: /"])
            return rp
        if resp.status_code >= 400:
            return None

        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp
