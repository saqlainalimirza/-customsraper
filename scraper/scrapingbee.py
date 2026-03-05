import re
import time
from urllib.parse import urlparse

import httpx

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

SCRAPINGBEE_API_URL = "https://app.scrapingbee.com/api/v1/"


class ScrapingBeeScraper:
    """Fallback scraper that fetches a site's main page via ScrapingBee."""

    def __init__(self):
        self.settings = get_settings()

    @staticmethod
    def _normalize_site_url(domain_or_url: str) -> str:
        if domain_or_url.startswith(("http://", "https://")):
            parsed = urlparse(domain_or_url)
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return f"https://{domain_or_url}".rstrip("/")

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def scrape_main_page(self, domain_or_url: str) -> tuple[str, str]:
        """
        Scrape only the root page for a domain/url.
        Returns (resolved_url, cleaned_text).
        """
        if not self.settings.scrapingbee_api_key:
            raise ValueError("SCRAPINGBEE_API_KEY is not configured")

        target_url = self._normalize_site_url(domain_or_url)
        candidates = [target_url]

        parsed = urlparse(target_url)
        if parsed.scheme == "https":
            candidates.append(f"http://{parsed.netloc}")
        if not parsed.netloc.startswith("www."):
            candidates.append(f"{parsed.scheme}://www.{parsed.netloc}")

        async with httpx.AsyncClient(timeout=self.settings.scrapingbee_timeout_seconds) as client:
            for url in candidates:
                started = time.perf_counter()
                try:
                    params = {
                        "api_key": self.settings.scrapingbee_api_key,
                        "url": url,
                        "render_js": "false",
                        "premium_proxy": "false",
                    }
                    response = await client.get(SCRAPINGBEE_API_URL, params=params)
                    duration_ms = (time.perf_counter() - started) * 1000
                    log_request(logger, "GET", url, response.status_code, duration_ms, {"provider": "scrapingbee"})
                    response.raise_for_status()

                    cleaned_text = self._clean_text(response.text)
                    if cleaned_text and len(cleaned_text) > 100:
                        return url, cleaned_text[:40000]
                    logger.warning(f"ScrapingBee returned low-content page for {url}: {len(cleaned_text)} chars")
                except Exception as exc:
                    logger.warning(f"ScrapingBee request failed for {url}: {exc}")

        raise ValueError(f"Failed to scrape main page via ScrapingBee for {domain_or_url}")
