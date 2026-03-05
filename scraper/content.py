import asyncio
import random
from typing import Dict

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]


class ContentScraper:
    def __init__(self):
        self.settings = get_settings()

    def _get_browser_config(self) -> BrowserConfig:
        """Create browser config with anti-blocking settings."""
        return BrowserConfig(
            headless=True,
            user_agent=random.choice(USER_AGENTS),
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            },
        )

    def _get_run_config(self) -> CrawlerRunConfig:
        return CrawlerRunConfig(
            wait_until="domcontentloaded",
            delay_before_return_html=0.3,
            remove_overlay_elements=False,
            page_timeout=15000,
        )

    async def _random_delay(self) -> None:
        """Add random delay between requests."""
        delay = random.uniform(
            self.settings.request_delay_min,
            self.settings.request_delay_max
        )
        logger.debug(f"Waiting {delay:.2f}s before next request")
        await asyncio.sleep(delay)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text content."""
        if not text:
            return ""
        
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                cleaned_lines.append(line)
        
        cleaned = "\n".join(cleaned_lines)
        
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")
        
        return cleaned.strip()

    async def scrape_urls(self, urls: list[str]) -> Dict[str, str]:
        """
        Scrape content from a list of URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dict mapping URL to extracted text content
        """
        logger.info(f"Starting to scrape {len(urls)} URLs")
        
        results: Dict[str, str] = {}
        
        async with AsyncWebCrawler(config=self._get_browser_config()) as crawler:
            for i, url in enumerate(urls):
                try:
                    log_request(logger, "GET", url)
                    result = await crawler.arun(url, config=self._get_run_config())
                    
                    if result.success:
                        text = result.markdown or result.cleaned_html or ""
                        
                        if not text and result.html:
                            import re
                            html = result.html
                            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                            html = re.sub(r'<[^>]+>', ' ', html)
                            html = re.sub(r'\s+', ' ', html)
                            text = html.strip()
                        
                        cleaned_text = self._clean_text(text)
                        
                        if cleaned_text and len(cleaned_text) > 100:
                            results[url] = cleaned_text[:30000]
                            logger.info(
                                f"[{i+1}/{len(urls)}] Scraped {url} | "
                                f"Content length: {len(cleaned_text)} chars"
                            )
                        else:
                            logger.warning(f"[{i+1}/{len(urls)}] No useful content from {url} (got {len(cleaned_text) if cleaned_text else 0} chars)")
                    else:
                        logger.warning(
                            f"[{i+1}/{len(urls)}] Failed to scrape {url}: {result.error_message}"
                        )
                        
                except Exception as e:
                    logger.error(f"[{i+1}/{len(urls)}] Error scraping {url}: {e}")
                    continue
        
        logger.info(f"Scraping complete. Successfully scraped {len(results)}/{len(urls)} URLs")
        
        return results

    async def scrape_single(self, url: str) -> str | None:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        results = await self.scrape_urls([url])
        return results.get(url)
