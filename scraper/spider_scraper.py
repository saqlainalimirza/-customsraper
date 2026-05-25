"""
spider.cloud scraper — the JS-rendering replacement for Jina on sites that
block / stall the simple reader. Smart mode auto-decides static vs browser.
Used by the /scrape/spider endpoint.

API: https://spider.cloud/docs/api
"""
import asyncio
import httpx
from urllib.parse import urlparse

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)


# Module-level cap so parallel rows don't burst Spider; shared across all instances.
# Env-tunable via SPIDER_CONCURRENCY.
_SPIDER_SEMAPHORE = asyncio.Semaphore(get_settings().spider_concurrency)


class SpiderScraper:
    """Thin async client for spider.cloud's /crawl endpoint."""

    def __init__(self):
        self.settings = get_settings()

    @staticmethod
    def _normalize_url(domain_or_url: str) -> str:
        s = (domain_or_url or "").strip()
        if not s:
            return ""
        if not s.startswith(("http://", "https://")):
            s = f"https://{s}"
        # strip trailing slash for cleanliness
        return s.rstrip("/")

    async def crawl_site(
        self,
        url: str,
        limit: int | None = None,
        mode: str = "smart",
        return_format: str = "markdown",
    ) -> dict[str, str]:
        """
        Crawl up to `limit` pages of a site and return {page_url: markdown}.
        Mode: "smart" (auto), "http" (no JS, fastest), or "chrome" (full browser).
        """
        if not self.settings.spider_api_key:
            raise ValueError("SPIDER_API_KEY is not set")

        url = self._normalize_url(url)
        if not url:
            return {}

        limit = limit or self.settings.spider_pages_per_site
        endpoint = f"{self.settings.spider_base_url.rstrip('/')}/crawl"
        headers = {
            "Authorization": f"Bearer {self.settings.spider_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "url": url,
            "limit": limit,
            "request": mode,
            "return_format": return_format,
            "proxy_enabled": True,
        }

        timeout = float(self.settings.spider_timeout_seconds)
        async with _SPIDER_SEMAPHORE:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                log_request(logger, "POST", endpoint, extra={"provider": "spider", "url": url, "limit": limit})
                try:
                    r = await client.post(endpoint, headers=headers, json=payload)
                    r.raise_for_status()
                except Exception as e:
                    logger.warning(f"[Spider] /crawl failed for {url}: {e}")
                    return {}

                try:
                    data = r.json()
                except Exception:
                    logger.warning(f"[Spider] non-JSON response for {url}: {r.text[:200]}")
                    return {}

        # Spider returns a list of page objects, or {"data": [...]}.
        items = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            logger.warning(f"[Spider] unexpected response shape for {url}: {str(data)[:200]}")
            return {}

        out: dict[str, str] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            page_url = it.get("url") or it.get("link") or ""
            content = it.get("content") or it.get("markdown") or it.get("html") or ""
            if not page_url or not content:
                continue
            # cap each page so a giant blog doesn't blow context
            out[page_url] = str(content)[:20000]

        logger.info(f"[Spider] {url} → {len(out)} pages crawled")
        return out

    async def scrape_url(
        self,
        url: str,
        mode: str = "smart",
        return_format: str = "markdown",
    ) -> dict[str, str]:
        """
        Single-page /scrape fallback for sites where /crawl returns nothing
        (anti-bot, no internal links, SPA root, etc.). Returns {url: markdown}
        in the same shape as crawl_site so callers can swap it in transparently.
        """
        if not self.settings.spider_api_key:
            raise ValueError("SPIDER_API_KEY is not set")

        url = self._normalize_url(url)
        if not url:
            return {}

        endpoint = f"{self.settings.spider_base_url.rstrip('/')}/scrape"
        headers = {
            "Authorization": f"Bearer {self.settings.spider_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "url": url,
            "request": mode,
            "return_format": return_format,
            "proxy_enabled": True,
        }

        timeout = float(self.settings.spider_timeout_seconds)
        async with _SPIDER_SEMAPHORE:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                log_request(logger, "POST", endpoint, extra={"provider": "spider-scrape", "url": url})
                try:
                    r = await client.post(endpoint, headers=headers, json=payload)
                    r.raise_for_status()
                except Exception as e:
                    logger.warning(f"[Spider] /scrape failed for {url}: {e}")
                    return {}

                try:
                    data = r.json()
                except Exception:
                    logger.warning(f"[Spider] /scrape non-JSON for {url}: {r.text[:200]}")
                    return {}

        items = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            logger.warning(f"[Spider] /scrape unexpected shape for {url}: {str(data)[:200]}")
            return {}

        out: dict[str, str] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            page_url = it.get("url") or it.get("link") or url
            content = it.get("content") or it.get("markdown") or it.get("html") or ""
            if not content:
                continue
            out[page_url] = str(content)[:20000]

        logger.info(f"[Spider] {url} → /scrape returned {len(out)} page(s)")
        return out
