import httpx
from urllib.parse import urlparse, quote

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"
JINA_SEARCH_BASE = "https://s.jina.ai/"


class JinaScraper:
    """
    Scrapes URLs via Jina Reader API (r.jina.ai).
    Returns clean, LLM-ready markdown content.
    Works without an API key (free tier: 20 RPM).
    Provide JINA_API_KEY for higher rate limits (500 RPM).
    """

    def __init__(self):
        self.settings = get_settings()

    def _build_headers(self) -> dict:
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "text",
        }
        if self.settings.jina_api_key:
            headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"
        return headers

    @staticmethod
    def _normalize_url(domain_or_url: str) -> str:
        if domain_or_url.startswith(("http://", "https://")):
            parsed = urlparse(domain_or_url)
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                url += f"?{parsed.query}"
            return url.rstrip("/")
        return f"https://{domain_or_url}".rstrip("/")

    async def scrape_url(self, url: str) -> str:
        """
        Fetch a single URL via Jina Reader and return clean text.
        Raises ValueError if content is insufficient.
        """
        jina_url = f"{JINA_READER_BASE}{url}"

        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            log_request(logger, "GET", jina_url, extra={"provider": "jina"})
            response = await client.get(jina_url, headers=self._build_headers())
            response.raise_for_status()

            content = response.text.strip()
            if not content or len(content) < 100:
                raise ValueError(
                    f"Jina returned insufficient content for {url}: {len(content)} chars"
                )

            logger.info(f"[Jina] Scraped {len(content)} chars from {url}")
            return content[:20000]

    async def scrape_main_page(self, domain_or_url: str) -> tuple[str, str]:
        """
        Scrape a domain's homepage via Jina Reader.
        Returns (resolved_url, clean_text).
        """
        url = self._normalize_url(domain_or_url)

        # Try https first, fall back to www variant
        candidates = [url]
        parsed = urlparse(url)
        if not parsed.netloc.startswith("www."):
            candidates.append(f"{parsed.scheme}://www.{parsed.netloc}")

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                content = await self.scrape_url(candidate)
                return candidate, content
            except Exception as exc:
                logger.warning(f"[Jina] Failed for {candidate}: {exc}")
                last_error = exc

        raise ValueError(
            f"Jina Reader failed to scrape {domain_or_url}: {last_error}"
        )

    async def search(self, query: str) -> list[dict]:
        """
        Search the web via Jina Search (s.jina.ai) and return up to 5 results.
        Each result: {url, title, content}
        """
        search_url = f"{JINA_SEARCH_BASE}?q={quote(query)}"
        headers = {
            "Accept": "application/json",
            "X-Respond-With": "no-content",  # Return metadata only (faster, no full page reads)
        }
        if self.settings.jina_api_key:
            headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            log_request(logger, "GET", search_url, extra={"provider": "jina-search"})
            response = await client.get(search_url, headers=headers)
            response.raise_for_status()

            # Jina Search returns JSON when Accept: application/json is set
            # Response shape: {"code": 200, "data": [{url, title, description, content}, ...]}
            try:
                data = response.json()
            except Exception:
                logger.warning(f"[Jina Search] Non-JSON response ({response.status_code}), raw: {response.text[:200]}")
                return []

            # Handle both {"data": [...]} and flat [...] response shapes
            items = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                logger.warning(f"[Jina Search] Unexpected response shape: {str(data)[:200]}")
                items = []

            results = [
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("description", item.get("content", "")),
                }
                for item in items
                if item.get("url")
            ]

            logger.info(f"[Jina Search] '{query[:60]}' → {len(results)} results: {[r['url'] for r in results]}")
            return results[:5]
