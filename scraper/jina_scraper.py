import re
import asyncio
import httpx
from urllib.parse import urlparse, urljoin, quote, parse_qsl, urlencode, urlunparse

from config import get_settings
from utils.logging import setup_logger, log_request

logger = setup_logger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"
JINA_SEARCH_BASE = "https://s.jina.ai/"

# Global throttles shared across ALL JinaScraper instances AND both endpoints
# (pipeline + agent). MUST live at module level so every row shares them. Sizes
# from settings (env-tunable: JINA_READER_CONCURRENCY / JINA_SEARCH_CONCURRENCY).
_jina_settings = get_settings()
_SEARCH_SEMAPHORE = asyncio.Semaphore(_jina_settings.jina_search_concurrency)  # s.jina.ai — tighter
_READER_SEMAPHORE = asyncio.Semaphore(_jina_settings.jina_reader_concurrency)  # r.jina.ai — generous

# Tracking params that create duplicate URLs (same page, different tag) — Google
# Shopping's srsltid is the big one in the logs; strip so we don't scrape a page
# 2-3× and waste Reader RPM.
_TRACKING_PARAMS = {
    "srsltid", "gclid", "fbclid", "msclkid", "dclid", "gclsrc", "_ga",
    "mc_cid", "mc_eid", "ref", "ref_src",
}


def strip_tracking_params(url: str) -> str:
    """Remove tracking query params (srsltid, utm_*, gclid, ...) for dedup/scrape."""
    try:
        parsed = urlparse(url)
        kept = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        return urlunparse(parsed._replace(query=urlencode(kept)))
    except Exception:
        return url


# Social/marketplace/aggregator domains that pollute search results — a wasted
# read. Filtered from EVERY search (pipeline + agent).
_JUNK_DOMAINS = (
    "instagram.com", "facebook.com", "youtube.com", "tiktok.com", "pinterest.com",
    "reddit.com", "linkedin.com", "twitter.com", "x.com", "amazon.com",
    "etsy.com", "ebay.com", "yelp.com", "wikipedia.org", "crunchbase.com",
)


def is_junk_url(url: str) -> bool:
    u = (url or "").lower()
    return any(d in u for d in _JUNK_DOMAINS)


def clean_search_query(q: str) -> str:
    """Strip operators that make Jina Search 422 or return junk: double-quotes,
    AND/OR, parentheses, site: filters. Plain words search better and never error."""
    if not q:
        return q
    q = q.replace('"', " ").replace("(", " ").replace(")", " ")
    q = re.sub(r"\bsite:\S+", " ", q, flags=re.I)
    q = re.sub(r"\b(OR|AND)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


class JinaScraper:
    """
    Scrapes URLs via Jina Reader API (r.jina.ai).
    Returns clean, LLM-ready markdown content.
    Works without an API key (free tier: 20 RPM).
    Provide JINA_API_KEY for higher rate limits (500 RPM).
    """

    def __init__(self):
        self.settings = get_settings()

    def _build_headers(self, keep_links: bool = False) -> dict:
        # Always return clean MARKDOWN (structured + far more compact than raw
        # text/HTML) and strip the heavy stuff that bloats tokens/memory:
        #   X-Retain-Images: none  → drop image markdown & data-URIs (huge saver)
        #   X-With-Images-Summary off, no links summary dump
        # keep_links is kept for call-site compatibility; markdown always carries
        # [text](url) links inline, which is what link discovery + the agent need.
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "markdown",
            "X-Retain-Images": "none",
            # Circuit breaker: pages bigger than the budget FAIL (caught upstream)
            # instead of flooding the LLM and blowing TPM.
            "X-Token-Budget": str(self.settings.jina_token_budget),
            "X-Timeout": str(self.settings.jina_timeout),
        }
        # Strip noise (videos/cookie banners/popups), keep nav + footer.
        if self.settings.jina_remove_selector:
            headers["X-Remove-Selector"] = self.settings.jina_remove_selector
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

    async def scrape_url(self, url: str, keep_links: bool = False) -> str:
        """
        Fetch a single URL via Jina Reader and return clean text.
        If keep_links=True, returns markdown with [text](url) links intact.
        Raises ValueError if content is insufficient.
        """
        url = strip_tracking_params(url)  # avoid scraping the same page under srsltid variants
        jina_url = f"{JINA_READER_BASE}{url}"

        _client_timeout = float(self.settings.jina_timeout) + 5.0  # above Jina's own timeout
        async with _READER_SEMAPHORE:
            async with httpx.AsyncClient(timeout=_client_timeout, follow_redirects=True) as client:
                log_request(logger, "GET", jina_url, extra={"provider": "jina"})
                response = await client.get(jina_url, headers=self._build_headers(keep_links=keep_links))
                response.raise_for_status()

                content = response.text.strip()
                if not content or len(content) < 100:
                    raise ValueError(
                        f"Jina returned insufficient content for {url}: {len(content)} chars"
                    )

                logger.info(f"[Jina] Scraped {len(content)} chars from {url} (keep_links={keep_links})")
                return content[:20000]

    async def scrape_main_page(self, domain_or_url: str, keep_links: bool = False) -> tuple[str, str]:
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
                content = await self.scrape_url(candidate, keep_links=keep_links)
                return candidate, content
            except Exception as exc:
                logger.warning(f"[Jina] Failed for {candidate}: {exc}")
                last_error = exc

        raise ValueError(
            f"Jina Reader failed to scrape {domain_or_url}: {last_error}"
        )

    @staticmethod
    def extract_links_from_markdown(markdown: str, base_url: str) -> list[dict]:
        """
        Parse [text](url) links from Jina markdown output. Returns same-domain
        links as [{text, url}], deduped, with asset/mailto/anchor links filtered out.
        """
        base_host = urlparse(base_url).netloc.lower().lstrip("www.")
        if not base_host:
            return []

        seen: set[str] = set()
        out: list[dict] = []
        # Match [text](url) — non-greedy, disallow nested brackets in text
        for match in re.finditer(r"\[([^\]]+)\]\(([^)\s]+)\)", markdown):
            text = match.group(1).strip()
            raw_url = match.group(2).strip()

            if not raw_url or raw_url.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue

            absolute = urljoin(base_url, raw_url)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue

            host = parsed.netloc.lower().lstrip("www.")
            if host != base_host:
                continue

            # Drop obvious asset links
            path_lower = parsed.path.lower()
            if path_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                                    ".pdf", ".zip", ".mp4", ".mp3", ".css", ".js",
                                    ".ico", ".woff", ".woff2", ".ttf")):
                continue

            # Normalise: strip fragment, trailing slash
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            if parsed.query:
                clean += f"?{parsed.query}"

            if clean in seen or clean.rstrip("/") == base_url.rstrip("/"):
                continue
            seen.add(clean)
            out.append({"text": text[:120], "url": clean})

        return out

    async def search(self, query: str) -> list[dict]:
        """
        Search the web via Jina Search (s.jina.ai) and return up to 5 results.
        Each result: {url, title, content}. Query is sanitized (no operators) and
        junk/social domains are filtered — applies to BOTH pipeline and agent.
        """
        query = clean_search_query(query)
        search_url = f"{JINA_SEARCH_BASE}?q={quote(query)}"
        headers = {
            "Accept": "application/json",
            "X-Respond-With": "no-content",  # Return metadata only (faster, no full page reads)
        }
        if self.settings.jina_api_key:
            headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"

        # httpx timeout sits just above Jina's own page-load timeout so the
        # client doesn't abort before Jina itself does.
        _client_timeout = float(self.settings.jina_timeout) + 5.0
        async with _SEARCH_SEMAPHORE:
          async with httpx.AsyncClient(timeout=_client_timeout, follow_redirects=True) as client:
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
                if item.get("url") and not is_junk_url(item.get("url", ""))
            ]

            logger.info(f"[Jina Search] '{query[:60]}' → {len(results)} results (junk filtered): {[r['url'] for r in results]}")
            return results[:5]
