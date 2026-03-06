import httpx
from urllib.parse import urlparse

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"


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
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return f"https://{domain_or_url}".rstrip("/")

    async def scrape_url(self, url: str) -> str:
        """
        Fetch a single URL via Jina Reader and return clean text.
        Raises ValueError if content is insufficient.
        """
        jina_url = f"{JINA_READER_BASE}{url}"

        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            log_request(logger, "GET", jina_url, extra={"provider": "jina"})
            response = await client.get(jina_url, headers=self._build_headers())
            response.raise_for_status()

            content = response.text.strip()
            if not content or len(content) < 100:
                raise ValueError(
                    f"Jina returned insufficient content for {url}: {len(content)} chars"
                )

            logger.info(f"[Jina] Scraped {len(content)} chars from {url}")
            return content[:40000]

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
