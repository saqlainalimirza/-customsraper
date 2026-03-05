import random
from urllib.parse import urlparse, urljoin
from typing import Set

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

MAX_URLS_TO_COLLECT = 30


class DomainCrawler:
    def __init__(self):
        self.settings = get_settings()
        self.discovered_urls: Set[str] = set()

    def _get_browser_config(self) -> BrowserConfig:
        """Create browser config with anti-blocking settings."""
        return BrowserConfig(
            headless=True,
            user_agent=random.choice(USER_AGENTS),
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
        )

    def _get_run_config(self) -> CrawlerRunConfig:
        """Create crawler run config - fast settings."""
        return CrawlerRunConfig(
            wait_until="domcontentloaded",
            delay_before_return_html=0.3,
            remove_overlay_elements=False,
            page_timeout=15000,
        )

    def _normalize_url(self, url: str, base_url: str) -> str | None:
        """Normalize and validate a URL."""
        if not url:
            return None
        
        if url.startswith(("#", "javascript:", "mailto:", "tel:")):
            return None
        
        full_url = urljoin(base_url, url)
        parsed = urlparse(full_url)
        
        if parsed.scheme not in ("http", "https"):
            return None
        
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized.rstrip("/")

    def _is_same_domain(self, url: str, domain: str) -> bool:
        """Check if URL belongs to the same domain."""
        parsed = urlparse(url)
        url_domain = parsed.netloc.lower()
        target_domain = domain.lower()
        return url_domain == target_domain or url_domain.endswith(f".{target_domain}")

    async def get_homepage_links(self, domain: str) -> list[str]:
        """
        Just get links from homepage - fast, single page crawl.
        Returns up to MAX_URLS_TO_COLLECT unique internal links.
        """
        start_url = f"https://{domain}"
        self.discovered_urls.clear()
        self.discovered_urls.add(start_url)
        
        logger.info(f"Getting links from homepage: {start_url}")
        
        async with AsyncWebCrawler(config=self._get_browser_config()) as crawler:
            try:
                log_request(logger, "GET", start_url)
                result = await crawler.arun(start_url, config=self._get_run_config())
                
                if not result.success:
                    logger.warning(f"Failed to crawl {start_url}, trying www version")
                    start_url = f"https://www.{domain}"
                    result = await crawler.arun(start_url, config=self._get_run_config())
                
                if result.success:
                    internal_links = result.links.get("internal", [])
                    
                    for link in internal_links:
                        if len(self.discovered_urls) >= MAX_URLS_TO_COLLECT:
                            break
                        href = link.get("href", "")
                        normalized = self._normalize_url(href, start_url)
                        if normalized and self._is_same_domain(normalized, domain):
                            self.discovered_urls.add(normalized)
                    
                    logger.info(f"Found {len(self.discovered_urls)} links from homepage")
                else:
                    logger.error(f"Failed to crawl homepage: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error crawling homepage {start_url}: {e}")
        
        return list(self.discovered_urls)[:MAX_URLS_TO_COLLECT]
