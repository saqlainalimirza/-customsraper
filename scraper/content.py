import asyncio
import random
from typing import Dict

import httpx
from bs4 import BeautifulSoup

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

    def _get_headers(self) -> dict:
        """Get HTTP headers with random user agent."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }

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
        Scrape content from a list of URLs using simple HTTP.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dict mapping URL to extracted text content
        """
        logger.info(f"Starting to scrape {len(urls)} URLs with simple HTTP")
        
        results: Dict[str, str] = {}
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for i, url in enumerate(urls):
                try:
                    log_request(logger, "GET", url)
                    response = await client.get(url, headers=self._get_headers())
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script, style, nav, footer, header elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                        element.decompose()
                    
                    # Get text from main content areas
                    text = soup.get_text()
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 100:
                        # Limit to 20k chars to avoid token limits
                        results[url] = cleaned_text[:20000]
                        logger.info(
                            f"[{i+1}/{len(urls)}] ✓ Scraped {url} | "
                            f"Content: {len(cleaned_text)} chars"
                        )
                    else:
                        logger.warning(f"[{i+1}/{len(urls)}] ✗ No useful content from {url} (got {len(cleaned_text) if cleaned_text else 0} chars)")
                        
                except Exception as e:
                    logger.warning(f"[{i+1}/{len(urls)}] ✗ Error scraping {url}: {e}")
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
