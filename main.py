import json
import re
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config import get_settings
from db.supabase_client import SupabaseClient
from scraper.crawler import DomainCrawler
from scraper.content import ContentScraper
from scraper.scrapingbee import ScrapingBeeScraper
from scraper.jina_scraper import JinaScraper, strip_tracking_params
from ai.openrouter_client import OpenRouterClient
from ai.base import AIClient
from utils.logging import setup_logger, log_pipeline_step, log_summary

logger = setup_logger(__name__)


def strip_json(text: str) -> str:
    """Strip markdown code fences like ```json ... ``` or ``` ... ``` from AI responses."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    return text.strip()


PARALLEL_WORKERS = 30
# No browser semaphore needed - using simple HTTP now!
FALLBACK_WORKERS = 10
ROW_TIMEOUT = 180
BATCH_DELAY = 2.0  # Delay between batches to prevent rate limits


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Web Scraper API")
    _s = get_settings()
    logger.info(
        f"[startup] resolved models → gemini={_s.gemini_model} | gpt={_s.gpt_model} | "
        f"claude={_s.claude_model} | default_provider={_s.default_ai_provider}"
    )
    yield
    logger.info("Shutting down AI Web Scraper API")


app = FastAPI(
    title="AI Web Scraper API",
    description="Scrape websites with AI-powered URL filtering and content extraction",
    version="1.0.0",
    lifespan=lifespan,
)


class ScrapeRequest(BaseModel):
    dataset_id: str
    prompt_filter: str
    prompt_extract: str
    limit: int = 100
    ai_provider: Literal["gpt", "claude"] = "gpt"
    run_fallback: bool = True
    fallback_limit: int | None = None


class ScrapeResponse(BaseModel):
    processed: int
    successful: int
    failed: int
    total_tokens: int
    fallback_processed: int = 0
    fallback_successful: int = 0
    fallback_failed: int = 0


class FallbackScrapeRequest(BaseModel):
    dataset_id: str
    prompt_extract: str
    limit: int = 100
    ai_provider: Literal["gpt", "claude"] = "gpt"


class SingleScrapeRequest(BaseModel):
    domain: str
    prompt_filter: str
    prompt_extract: str
    ai_provider: Literal["gpt", "claude"] = "gpt"


class SingleScrapeResponse(BaseModel):
    all_urls: list[str]
    filtered_urls: list[str]
    scraped_content: dict[str, str]
    extracted_answer: str
    filter_input_tokens: int
    filter_output_tokens: int
    extract_input_tokens: int
    extract_output_tokens: int
    total_tokens: int


class DirectScrapeRequest(BaseModel):
    url: str
    prompt_filter: str
    prompt_extract: str
    ai_provider: Literal["gpt", "claude"] = "gpt"


class JinaSmartRequest(BaseModel):
    # dict[str, Any] (not str) so Clay can send nulls / numbers / blanks for a
    # field without a 422. We coerce + drop empties in normalize().
    data: dict[str, Any] | None = None
    url: str | None = None  # Top-level fallback for Clay integration
    website: str | None = None  # Top-level fallback
    prompt_extract: str | None = None
    prompt_filter: str | None = None
    # Free-form so the agent can accept aliases ("flashlite", "grok", "mix", ...).
    # Pipeline endpoint only understands gpt/claude/gemini (falls back to gemini).
    ai_provider: str = "gemini"

    def normalize(self) -> "JinaSmartRequest":
        """
        Merge top-level fields into data, then clean the data dict:
          - coerce every value to a trimmed string
          - DROP keys whose value is missing/null/empty/"null"/"undefined"
        So Clay can send a field name with no value and we just ignore it
        instead of breaking or feeding garbage to the AI.
        """
        raw = self.data or {}
        if self.url and "url" not in raw:
            raw["url"] = self.url
        if self.website and "website" not in raw:
            raw["website"] = self.website

        cleaned: dict[str, str] = {}
        for k, v in raw.items():
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() in ("null", "undefined", "none", "nan", "n/a"):
                continue
            cleaned[k] = s
        self.data = cleaned
        return self


class DirectScrapeResponse(BaseModel):
    url: str
    all_urls: list[str]
    filtered_urls: list[str]
    scraped_content: dict[str, str]
    extracted_answer: Any  # parsed JSON dict, or "NOTFOUND" string
    filter_input_tokens: int
    filter_output_tokens: int
    extract_input_tokens: int
    extract_output_tokens: int
    total_tokens: int


def get_ai_client(provider: str) -> AIClient:
    # Any alias/mix string — resolved via the shared model registry inside the client.
    return OpenRouterClient(model_type=provider or "gemini")


def extract_domain(domain_or_url: str) -> str:
    if domain_or_url.startswith(("http://", "https://")):
        parsed = urlparse(domain_or_url)
        return parsed.netloc
    return domain_or_url.replace("www.", "")


async def process_single_row(
    row: dict,
    prompt_filter: str,
    prompt_extract: str,
    ai_client: AIClient,
    db: SupabaseClient,
) -> dict:
    """Process a single scrape job row with timeout and browser limiting."""
    row_id = row["id"]
    domain = extract_domain(row["domain"])
    
    result = {
        "all_urls": [],
        "filtered_urls": [],
        "scraped_content": {},
        "extracted_answer": "",
        "filter_input_tokens": 0,
        "filter_output_tokens": 0,
        "extract_input_tokens": 0,
        "extract_output_tokens": 0,
    }
    
    try:
        # Wrap entire row in a timeout
        return await asyncio.wait_for(
            _do_process_row(row_id, domain, prompt_filter, prompt_extract, ai_client, db, result),
            timeout=ROW_TIMEOUT,
        )
    except asyncio.TimeoutError:
        error_msg = f"Timed out after {ROW_TIMEOUT}s"
        logger.error(f"Row {row_id} ({domain}): {error_msg}")
        await db.mark_failed(row_id, error_msg)
        raise
    except Exception as e:
        error_msg = str(e)
        log_pipeline_step(logger, "process", row_id, "failed", {"error": error_msg})
        await db.mark_failed(row_id, error_msg)
        raise


async def _do_process_row(row_id, domain, prompt_filter, prompt_extract, ai_client, db, result):
    """Core processing logic - using simple HTTP (no browser needed!)."""
    
    # STEP 1: Crawl homepage with simple HTTP
    crawler = DomainCrawler()
    all_urls = await crawler.get_homepage_links(domain)
    
    result["all_urls"] = all_urls
    if not all_urls:
        raise ValueError(f"No URLs found for domain {domain}")
    
    logger.info(f"[{domain}] Found {len(all_urls)} links")
    
    # STEP 2: AI filter URLs
    filter_response = await ai_client.filter_urls(all_urls, prompt_filter, domain)
    result["filter_input_tokens"] = filter_response.input_tokens
    result["filter_output_tokens"] = filter_response.output_tokens
    
    try:
        filtered_urls = json.loads(filter_response.content)
        if not isinstance(filtered_urls, list):
            filtered_urls = []
        filtered_urls = filtered_urls[:5]
    except json.JSONDecodeError:
        filtered_urls = all_urls[:5]
    
    result["filtered_urls"] = filtered_urls
    if not filtered_urls:
        raise ValueError("AI filtered out all URLs")
    
    logger.info(f"[{domain}] AI picked {len(filtered_urls)} URLs")
    
    # STEP 3: Scrape filtered URLs with simple HTTP
    content_scraper = ContentScraper()
    scraped_content = await content_scraper.scrape_urls(filtered_urls)
    
    result["scraped_content"] = scraped_content
    if not scraped_content:
        raise ValueError("Failed to scrape any content")
    
    logger.info(f"[{domain}] Scraped {len(scraped_content)} pages")
    
    # STEP 4: AI extract answer
    extract_response = await ai_client.extract_answer(scraped_content, prompt_extract)
    result["extracted_answer"] = extract_response.content
    result["extract_input_tokens"] = extract_response.input_tokens
    result["extract_output_tokens"] = extract_response.output_tokens
    
    logger.info(f"[{domain}] Done! Answer: {len(extract_response.content)} chars")
    
    # STEP 5: Save to DB
    await db.mark_completed(
        row_id=row_id,
        all_urls=result["all_urls"],
        filtered_urls=result["filtered_urls"],
        scraped_content=result["scraped_content"],
        extracted_answer=result["extracted_answer"],
        filter_input_tokens=result["filter_input_tokens"],
        filter_output_tokens=result["filter_output_tokens"],
        extract_input_tokens=result["extract_input_tokens"],
        extract_output_tokens=result["extract_output_tokens"],
    )
    
    return result


async def process_fallback_row(
    row: dict,
    prompt_extract: str,
    ai_client: AIClient,
    db: SupabaseClient,
) -> dict:
    """
    Fallback row processing:
    - Skip URL discovery
    - Skip URL filtering prompt
    - Scrape only main page with ScrapingBee
    - Run extract prompt directly
    """
    row_id = row["id"]
    domain = extract_domain(row["domain"])

    try:
        await db.update_status(row_id, "fallback_scraping")
        scrapingbee = ScrapingBeeScraper()
        resolved_url, main_page_content = await scrapingbee.scrape_main_page(domain)

        scraped_content = {resolved_url: main_page_content}
        await db.update_status(row_id, "fallback_extracting")
        extract_response = await ai_client.extract_answer(scraped_content, prompt_extract)

        await db.mark_completed(
            row_id=row_id,
            all_urls=[resolved_url],
            filtered_urls=[resolved_url],
            scraped_content=scraped_content,
            extracted_answer=extract_response.content,
            filter_input_tokens=0,
            filter_output_tokens=0,
            extract_input_tokens=extract_response.input_tokens,
            extract_output_tokens=extract_response.output_tokens,
        )

        return {
            "extract_input_tokens": extract_response.input_tokens,
            "extract_output_tokens": extract_response.output_tokens,
        }
    except Exception as e:
        error_msg = f"Fallback pipeline failed: {e}"
        log_pipeline_step(logger, "fallback_process", row_id, "failed", {"error": error_msg})
        await db.mark_failed(row_id, error_msg)
        raise


async def run_fallback_pipeline(
    db: SupabaseClient,
    ai_client: AIClient,
    dataset_id: str,
    prompt_extract: str,
    limit: int,
) -> tuple[int, int, int, int]:
    """Run fallback pipeline for failed rows in a dataset."""
    failed_rows = await db.get_failed(dataset_id, limit)
    if not failed_rows:
        logger.info(f"No failed rows found for fallback in dataset_id={dataset_id}")
        return 0, 0, 0, 0

    fallback_successful = 0
    fallback_failed = 0
    fallback_tokens = 0

    for i in range(0, len(failed_rows), FALLBACK_WORKERS):
        batch = failed_rows[i:i + FALLBACK_WORKERS]
        logger.info(f"=== Fallback Batch {i//FALLBACK_WORKERS + 1}: {len(batch)} rows ===")

        tasks = [process_fallback_row(row, prompt_extract, ai_client, db) for row in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Fallback failed row {batch[idx]['id']}: {result}")
                fallback_failed += 1
            else:
                fallback_successful += 1
                fallback_tokens += result["extract_input_tokens"] + result["extract_output_tokens"]
        
        # Add delay between fallback batches to avoid ScrapingBee rate limits
        if i + FALLBACK_WORKERS < len(failed_rows):
            await asyncio.sleep(BATCH_DELAY)

    return len(failed_rows), fallback_successful, fallback_failed, fallback_tokens


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_batch(request: ScrapeRequest):
    """Process a batch of scrape jobs in parallel (50 rows, 15 browsers max)."""
    logger.info(
        f"Starting batch scrape for dataset_id={request.dataset_id}, limit={request.limit} | "
        f"prompt_filter: {request.prompt_filter[:50]}... | prompt_extract: {request.prompt_extract[:50]}..."
    )
    
    db = SupabaseClient()
    ai_client = get_ai_client(request.ai_provider)
    
    rows = await db.get_unprocessed(request.dataset_id, request.limit)
    
    if not rows:
        logger.info(f"No unprocessed rows found for dataset_id={request.dataset_id}")
        return ScrapeResponse(
            processed=0,
            successful=0,
            failed=0,
            total_tokens=0,
            fallback_processed=0,
            fallback_successful=0,
            fallback_failed=0,
        )
    
    successful = 0
    failed = 0
    total_tokens = 0
    
    for i in range(0, len(rows), PARALLEL_WORKERS):
        batch = rows[i:i + PARALLEL_WORKERS]
        logger.info(f"=== Batch {i//PARALLEL_WORKERS + 1}: {len(batch)} rows ===")
        
        tasks = [
            process_single_row(row, request.prompt_filter, request.prompt_extract, ai_client, db)
            for row in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed row {batch[idx]['id']}: {result}")
                failed += 1
            else:
                successful += 1
                total_tokens += (
                    result["filter_input_tokens"] + result["filter_output_tokens"] +
                    result["extract_input_tokens"] + result["extract_output_tokens"]
                )
        
        # Add delay between batches to prevent overwhelming resources
        if i + PARALLEL_WORKERS < len(rows):
            await asyncio.sleep(BATCH_DELAY)

    fallback_processed = 0
    fallback_successful = 0
    fallback_failed = 0

    if request.run_fallback:
        fallback_limit = request.fallback_limit or request.limit
        (
            fallback_processed,
            fallback_successful,
            fallback_failed,
            fallback_tokens,
        ) = await run_fallback_pipeline(
            db=db,
            ai_client=ai_client,
            dataset_id=request.dataset_id,
            prompt_extract=request.prompt_extract,
            limit=fallback_limit,
        )
        total_tokens += fallback_tokens
    
    log_summary(logger, request.dataset_id, len(rows), successful, failed, total_tokens)
    
    return ScrapeResponse(
        processed=len(rows),
        successful=successful,
        failed=failed,
        total_tokens=total_tokens,
        fallback_processed=fallback_processed,
        fallback_successful=fallback_successful,
        fallback_failed=fallback_failed,
    )


@app.post("/scrape/fallback", response_model=ScrapeResponse)
async def scrape_failed_rows_with_fallback(request: FallbackScrapeRequest):
    """Re-process failed rows using ScrapingBee main-page scraping + extract prompt only."""
    logger.info(
        f"Starting fallback-only scrape for dataset_id={request.dataset_id}, limit={request.limit} | "
        f"prompt_extract: {request.prompt_extract[:50]}..."
    )

    db = SupabaseClient()
    ai_client = get_ai_client(request.ai_provider)

    (
        fallback_processed,
        fallback_successful,
        fallback_failed,
        fallback_tokens,
    ) = await run_fallback_pipeline(
        db=db,
        ai_client=ai_client,
        dataset_id=request.dataset_id,
        prompt_extract=request.prompt_extract,
        limit=request.limit,
    )

    return ScrapeResponse(
        processed=0,
        successful=0,
        failed=0,
        total_tokens=fallback_tokens,
        fallback_processed=fallback_processed,
        fallback_successful=fallback_successful,
        fallback_failed=fallback_failed,
    )


@app.post("/scrape/single", response_model=SingleScrapeResponse)
async def scrape_single(request: SingleScrapeRequest):
    """Process a single scrape request without Supabase."""
    logger.info(f"Starting single scrape for domain={request.domain}")
    
    ai_client = get_ai_client(request.ai_provider)
    domain = extract_domain(request.domain)
    
    crawler = DomainCrawler()
    all_urls = await crawler.get_homepage_links(domain)
    
    if not all_urls:
        raise HTTPException(status_code=404, detail=f"No URLs found for domain {domain}")
    
    filter_response = await ai_client.filter_urls(all_urls, request.prompt_filter, domain)
    
    try:
        filtered_urls = json.loads(filter_response.content)
        if not isinstance(filtered_urls, list):
            filtered_urls = []
    except json.JSONDecodeError:
        filtered_urls = all_urls[:5]
    
    if not filtered_urls:
        raise HTTPException(status_code=404, detail="No relevant URLs found after filtering")
    
    content_scraper = ContentScraper()
    scraped_content = await content_scraper.scrape_urls(filtered_urls)
    
    if not scraped_content:
        raise HTTPException(status_code=500, detail="Failed to scrape content from URLs")
    
    extract_response = await ai_client.extract_answer(scraped_content, request.prompt_extract)
    
    total_tokens = (
        filter_response.input_tokens + filter_response.output_tokens +
        extract_response.input_tokens + extract_response.output_tokens
    )
    
    return SingleScrapeResponse(
        all_urls=all_urls,
        filtered_urls=filtered_urls,
        scraped_content=scraped_content,
        extracted_answer=extract_response.content,
        filter_input_tokens=filter_response.input_tokens,
        filter_output_tokens=filter_response.output_tokens,
        extract_input_tokens=extract_response.input_tokens,
        extract_output_tokens=extract_response.output_tokens,
        total_tokens=total_tokens,
    )


@app.post("/scrape/direct", response_model=DirectScrapeResponse)
async def scrape_direct_url(request: DirectScrapeRequest):
    """
    Process a single URL directly without any database interaction.
    Takes a URL, scrapes it, and returns the extracted information.
    If data not found, automatically falls back to ScrapingBee for deeper scraping.
    """
    logger.info(f"Starting direct scrape for URL={request.url}")
    
    try:
        ai_client = get_ai_client(request.ai_provider)
        domain = extract_domain(request.url)
        
        # STEP 1: Crawl homepage to get all links
        crawler = DomainCrawler()
        all_urls = await crawler.get_homepage_links(domain)
        
        if not all_urls:
            raise HTTPException(status_code=404, detail=f"No URLs found for domain {domain}")
        
        logger.info(f"[{domain}] Found {len(all_urls)} links")
        
        # STEP 2: AI filter URLs based on prompt
        filter_response = await ai_client.filter_urls(all_urls, request.prompt_filter, domain)
        
        try:
            filtered_urls = json.loads(filter_response.content)
            if not isinstance(filtered_urls, list):
                filtered_urls = []
            filtered_urls = filtered_urls[:5]  # Limit to top 5
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI filter response, using first 5 URLs")
            filtered_urls = all_urls[:5]
        
        if not filtered_urls:
            raise HTTPException(status_code=404, detail="No relevant URLs found after AI filtering")
        
        logger.info(f"[{domain}] AI picked {len(filtered_urls)} URLs")
        
        # STEP 3: Scrape content from filtered URLs
        content_scraper = ContentScraper()
        scraped_content = await content_scraper.scrape_urls(filtered_urls)
        
        if not scraped_content:
            raise HTTPException(status_code=500, detail="Failed to scrape content from any URLs")
        
        logger.info(f"[{domain}] Scraped {len(scraped_content)} pages")
        
        # STEP 4: AI extract answer from scraped content
        extract_response = await ai_client.extract_answer(scraped_content, request.prompt_extract)
        
        logger.info(f"[{domain}] Extraction complete: {len(extract_response.content)} chars")
        
        # Initialize token tracking
        filter_input_tokens = filter_response.input_tokens
        filter_output_tokens = filter_response.output_tokens
        extract_input_tokens = extract_response.input_tokens
        extract_output_tokens = extract_response.output_tokens
        extracted_answer = extract_response.content
        
        # STEP 5: Fallback chain if data not found — ScrapingBee → Jina Reader
        if extracted_answer.strip().upper() == "NOTFOUND":
            logger.warning(f"[{domain}] Data not found with regular scraping, falling back to ScrapingBee")
            scrapingbee_ok = False

            try:
                scrapingbee = ScrapingBeeScraper()
                resolved_url, main_page_content = await scrapingbee.scrape_main_page(domain)
                scrapingbee_scraped = {resolved_url: main_page_content}
                sb_extract = await ai_client.extract_answer(scrapingbee_scraped, request.prompt_extract)

                scraped_content = scrapingbee_scraped
                filtered_urls = [resolved_url]
                extracted_answer = sb_extract.content
                extract_input_tokens += sb_extract.input_tokens
                extract_output_tokens += sb_extract.output_tokens
                scrapingbee_ok = True
                logger.info(f"[{domain}] ScrapingBee fallback successful")

            except Exception as sb_error:
                logger.error(f"[{domain}] ScrapingBee fallback failed: {sb_error}")

            # Jina fallback: triggered if ScrapingBee failed OR still returned NOTFOUND
            if not scrapingbee_ok or extracted_answer.strip().upper() == "NOTFOUND":
                logger.warning(f"[{domain}] Falling back to Jina Reader")
                try:
                    jina = JinaScraper()
                    jina_url, jina_content = await jina.scrape_main_page(domain)
                    jina_scraped = {jina_url: jina_content}
                    jina_extract = await ai_client.extract_answer(jina_scraped, request.prompt_extract)

                    scraped_content = jina_scraped
                    filtered_urls = [jina_url]
                    extracted_answer = jina_extract.content
                    extract_input_tokens += jina_extract.input_tokens
                    extract_output_tokens += jina_extract.output_tokens
                    logger.info(f"[{domain}] Jina fallback successful")

                except Exception as jina_error:
                    logger.error(f"[{domain}] Jina fallback failed: {jina_error}")
                    # Keep whatever result we have at this point

        # Parse extracted_answer JSON into a dict when possible
        parsed_answer: Any = extracted_answer
        if extracted_answer.strip().upper() != "NOTFOUND":
            try:
                parsed_answer = json.loads(strip_json(extracted_answer))
            except (json.JSONDecodeError, ValueError):
                parsed_answer = extracted_answer  # Return as raw string if not valid JSON

        # Calculate total tokens
        total_tokens = (
            filter_input_tokens + filter_output_tokens +
            extract_input_tokens + extract_output_tokens
        )

        return DirectScrapeResponse(
            url=request.url,
            all_urls=all_urls,
            filtered_urls=filtered_urls,
            scraped_content=scraped_content,
            extracted_answer=parsed_answer,
            filter_input_tokens=filter_input_tokens,
            filter_output_tokens=filter_output_tokens,
            extract_input_tokens=extract_input_tokens,
            extract_output_tokens=extract_output_tokens,
            total_tokens=total_tokens,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing direct scrape: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/scrape/scrapingbee-only", response_model=ScrapeResponse)
async def scrape_with_scrapingbee_only(request: FallbackScrapeRequest):
    """
    Process UNPROCESSED rows using ONLY ScrapingBee (no Playwright, no URL filtering).
    - Fetch unprocessed rows
    - Scrape main page with ScrapingBee
    - Extract with AI
    - Fast and stable, bypasses all browser issues
    """
    logger.info(
        f"Starting ScrapingBee-only scrape for dataset_id={request.dataset_id}, limit={request.limit} | "
        f"prompt_extract: {request.prompt_extract[:50]}..."
    )
    
    db = SupabaseClient()
    ai_client = get_ai_client(request.ai_provider)
    
    # Get UNPROCESSED rows (not failed ones)
    rows = await db.get_unprocessed(request.dataset_id, request.limit)
    
    if not rows:
        logger.info(f"No unprocessed rows found for dataset_id={request.dataset_id}")
        return ScrapeResponse(
            processed=0,
            successful=0,
            failed=0,
            total_tokens=0,
            fallback_processed=0,
            fallback_successful=0,
            fallback_failed=0,
        )
    
    successful = 0
    failed = 0
    total_tokens = 0
    
    for i in range(0, len(rows), FALLBACK_WORKERS):
        batch = rows[i:i + FALLBACK_WORKERS]
        logger.info(f"=== ScrapingBee Batch {i//FALLBACK_WORKERS + 1}: {len(batch)} rows ===")
        
        tasks = [process_fallback_row(row, request.prompt_extract, ai_client, db) for row in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"ScrapingBee failed row {batch[idx]['id']}: {result}")
                failed += 1
            else:
                successful += 1
                total_tokens += result["extract_input_tokens"] + result["extract_output_tokens"]
        
        # Add delay between batches to avoid ScrapingBee rate limits
        if i + FALLBACK_WORKERS < len(rows):
            await asyncio.sleep(BATCH_DELAY)
    
    log_summary(logger, request.dataset_id, len(rows), successful, failed, total_tokens)
    
    return ScrapeResponse(
        processed=len(rows),
        successful=successful,
        failed=failed,
        total_tokens=total_tokens,
        fallback_processed=0,
        fallback_successful=0,
        fallback_failed=0,
    )


@app.post("/scrape/jina-test")
async def scrape_jina_test(request: JinaSmartRequest):
    """
    Two-track Jina pipeline (both tracks run in parallel, readers run in parallel):

    Track A — Direct website scrape via Jina Reader
    Track B — AI search query → Jina Search → parallel Jina Reader on top 3 results

    Combined content → final AI extraction.
    Attempt 1 is fast (35s cap). The retry is generous (85s cap) — like a fresh
    manual rerun. Typical success ~20-25s; worst case (slow row, full retry) ~124s.
    """
    request.normalize()
    logger.info(f"[Jina Smart] Starting pipeline with data keys: {list(request.data.keys())}")

    async def _run(track_a_deadline: float, track_b_deadline: float, max_queries: int = 2) -> dict:
        ai_client = get_ai_client(request.ai_provider)
        jina = JinaScraper()

        # Allow prompts nested inside data (Clay/Zapier style)
        prompt_extract = request.prompt_extract or request.data.get("prompt_extract", "")
        if not prompt_extract:
            raise HTTPException(status_code=422, detail="prompt_extract is required (top-level or inside data)")

        # Strip prompt keys — only company fields go to AI for query generation
        PROMPT_KEYS = {"prompt_extract", "prompt_filter"}
        clean_data = {k: v for k, v in request.data.items() if k not in PROMPT_KEYS}

        # Robust website detection: Clay maps the URL to all sorts of column
        # names. Check the common ones first, then fall back to ANY value that
        # looks like a URL/domain so a non-standard field name still works.
        WEBSITE_KEYS = (
            "website", "url", "domain", "website_url", "site", "homepage",
            "company_website", "web", "company_url", "company_domain",
        )
        website_url = ""
        for key in WEBSITE_KEYS:
            if clean_data.get(key):
                website_url = clean_data[key].strip()
                break
        if not website_url:
            # Fallback: scan every value for something domain-shaped
            for v in clean_data.values():
                s = str(v).strip()
                if s.startswith(("http://", "https://")) or re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", s, re.I):
                    if "linkedin.com" in s.lower():
                        continue  # skip the LinkedIn URL — not the company site
                    website_url = s
                    logger.info(f"[Jina Smart] Auto-detected website from unlabeled field: {s}")
                    break

        # ── TRACK A: Direct website scrape + 1-hop agentic link discovery ────
        # Goal: extraction goals like "case studies" live on subpages with
        # unpredictable names (/work, /portfolio, /what-we-do, ...). We fetch
        # the homepage in markdown (links intact), let the LLM pick up to 3
        # relevant internal links, and Jina-read those in parallel.
        async def run_track_a() -> tuple[dict[str, str], list[str]]:
            if not website_url:
                logger.warning("[Jina Smart][Track A] No website provided, skipping")
                return {}, []
            try:
                homepage_url, homepage_md = await jina.scrape_main_page(website_url, keep_links=True)
                logger.info(f"[Jina Smart][Track A] Homepage: {len(homepage_md)} chars from {homepage_url}")
            except Exception as e:
                logger.warning(f"[Jina Smart][Track A] Homepage fetch failed for {website_url}: {e}")
                return {}, []

            result: dict[str, str] = {homepage_url: homepage_md}
            picked_urls: list[str] = []

            # Only attempt link discovery if the AI client supports it (OpenRouter)
            if not hasattr(ai_client, "pick_relevant_links"):
                return result, picked_urls

            try:
                links = jina.extract_links_from_markdown(homepage_md, homepage_url)
                logger.info(f"[Jina Smart][Track A] Extracted {len(links)} internal links from homepage")
                if not links:
                    return result, picked_urls

                pick_response = await ai_client.pick_relevant_links(
                    links=links,
                    prompt_extract=prompt_extract,
                    homepage_url=homepage_url,
                    max_links=4,
                )
                try:
                    picked_urls = json.loads(pick_response.content).get("urls", []) or []
                except (json.JSONDecodeError, AttributeError):
                    picked_urls = []

                if not picked_urls:
                    logger.info("[Jina Smart][Track A] No relevant subpages identified")
                    return result, picked_urls

                # Fetch picked subpages in parallel (plain text, not markdown)
                subpage_tasks = [jina.scrape_url(u) for u in picked_urls]
                subpage_results = await asyncio.gather(*subpage_tasks, return_exceptions=True)
                for url, res in zip(picked_urls, subpage_results):
                    if isinstance(res, Exception):
                        logger.warning(f"[Jina Smart][Track A] Subpage fetch failed for {url}: {res}")
                        continue
                    result[url] = res
                    logger.info(f"[Jina Smart][Track A] Subpage: {len(res)} chars from {url}")
            except Exception as e:
                logger.warning(f"[Jina Smart][Track A] Link discovery failed (keeping homepage only): {e}")

            return result, picked_urls

        # ── TRACK B: Search → parallel read ───────────────────────────────────
        async def run_track_b() -> tuple[str, list[dict], dict[str, str]]:
            try:
                query_response = await ai_client.generate_search_query(clean_data, prompt_extract)
                try:
                    queries = json.loads(query_response.content)
                    if not isinstance(queries, list):
                        queries = [str(queries)]
                except (json.JSONDecodeError, TypeError):
                    queries = [query_response.content.strip()]
                queries = [q.strip() for q in queries if q and q.strip()][:max_queries]
                search_query = " | ".join(queries)  # for logging/response display
                logger.info(f"[Jina Smart][Track B] {len(queries)} search queries (cap {max_queries}): {queries}")
                if not queries:
                    return "", [], {}

                # Run all queries in parallel, then merge + dedupe results by URL
                search_tasks = [jina.search(q) for q in queries]
                per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
                merged: list[dict] = []
                seen_urls: set[str] = set()
                for q, res in zip(queries, per_query):
                    if isinstance(res, Exception):
                        logger.warning(f"[Jina Smart][Track B] Search failed for '{q}': {res}")
                        continue
                    for item in res:
                        u = item.get("url")
                        clean = strip_tracking_params(u) if u else u
                        if u and clean not in seen_urls:
                            seen_urls.add(clean)
                            merged.append(item)
                search_results = merged

                if not search_results:
                    logger.warning(f"[Jina Smart][Track B] No results for any of: {queries}")
                    return search_query, [], {}

                raw_search_results = [
                    {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")}
                    for r in search_results
                ]
                logger.info(f"[Jina Smart][Track B] {len(raw_search_results)} merged results: {[r['url'] for r in raw_search_results]}")

                # Actually SCRAPE the top 2 search results — passing only snippets
                # to the LLM was the #1 cause of false NOTFOUND. Snippet ≠ page.
                top_urls = [r["url"] for r in search_results[:2] if r.get("url")]
                snippet_lookup = {
                    r["url"]: f"[Title]: {r.get('title', '')}\n[Snippet]: {r.get('content', '')}"
                    for r in search_results if r.get("url")
                }
                content_map: dict[str, str] = {}
                if top_urls:
                    scrape_tasks = [jina.scrape_url(u) for u in top_urls]
                    scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                    for url, res in zip(top_urls, scrape_results):
                        if isinstance(res, Exception):
                            logger.warning(f"[Jina Smart][Track B] Scrape failed for {url}: {res} — falling back to snippet")
                            content_map[url] = snippet_lookup.get(url, "")
                        else:
                            content_map[url] = res
                            logger.info(f"[Jina Smart][Track B] Scraped {len(res)} chars from {url}")
                # Remaining results (3rd, 4th, 5th) keep snippet-only — better than nothing
                for r in search_results[2:]:
                    url = r.get("url")
                    if url and url not in content_map:
                        content_map[url] = snippet_lookup.get(url, "")
                return search_query, raw_search_results, content_map

            except Exception as e:
                logger.warning(f"[Jina Smart][Track B] Track failed entirely (continuing with Track A only): {e}")
                return "", [], {}

        # ── Both tracks in parallel, each with its own per-track deadline ────
        # If Track A is slow but Track B finishes, we still extract from Track B
        # (and vice versa). Only fail if BOTH tracks return nothing.
        # Deadlines are passed in: the first run is fast; the retry is generous
        # (full time budget, like a manual rerun) since NOTFOUND-time is fine.
        async def _track_a_with_timeout():
            try:
                return await asyncio.wait_for(run_track_a(), timeout=track_a_deadline)
            except asyncio.TimeoutError:
                logger.warning(f"[Jina Smart][Track A] Hit {track_a_deadline}s deadline — using whatever Track B has")
                return {}, []

        async def _track_b_with_timeout():
            try:
                return await asyncio.wait_for(run_track_b(), timeout=track_b_deadline)
            except asyncio.TimeoutError:
                logger.warning(f"[Jina Smart][Track B] Hit {track_b_deadline}s deadline — using whatever Track A has")
                return "", [], {}

        (track_a_result, track_a_picked_urls), (search_query, raw_search_results, track_b_result) = await asyncio.gather(
            _track_a_with_timeout(),
            _track_b_with_timeout(),
        )

        # ── CROSS-POLLINATION: salvage same-domain URLs from search ─────────
        # Track A picks subpages from homepage links — but if the homepage
        # doesn't link to /shop or /products, those pages get missed. Google
        # (via Track B's search) usually DOES find them. If a search result is
        # on the company's own domain and Track A missed it, scrape it now.
        # This is the single biggest fix for "homepage info only" NOTFOUND rows.
        if website_url and raw_search_results:
            try:
                normalized = jina._normalize_url(website_url)
                company_host = urlparse(normalized).netloc.lower().lstrip("www.")
                # Dedupe by tracking-stripped URL so the same page under different
                # ?srsltid= tags isn't scraped 2-3× (was wasting Reader RPM).
                already_have = {strip_tracking_params(u) for u in track_a_result.keys()}
                seen_clean: set[str] = set()
                salvage_urls: list[str] = []
                for r in raw_search_results:
                    u = r.get("url")
                    if not u:
                        continue
                    clean = strip_tracking_params(u)
                    host = urlparse(clean).netloc.lower().lstrip("www.")
                    if host == company_host and clean not in already_have and clean not in seen_clean:
                        seen_clean.add(clean)
                        salvage_urls.append(clean)
                salvage_urls = salvage_urls[:3]
                if salvage_urls:
                    logger.info(f"[Jina Smart] Cross-poll: scraping {len(salvage_urls)} same-domain URLs from search: {salvage_urls}")
                    salvage_tasks = [jina.scrape_url(u) for u in salvage_urls]
                    salvage_results = await asyncio.gather(*salvage_tasks, return_exceptions=True)
                    for u, res in zip(salvage_urls, salvage_results):
                        if isinstance(res, Exception):
                            logger.warning(f"[Jina Smart] Cross-poll scrape failed for {u}: {res}")
                            continue
                        track_a_result[u] = res
                        track_a_picked_urls.append(u)
                        logger.info(f"[Jina Smart] Cross-poll: +{len(res)} chars from {u}")
            except Exception as e:
                logger.warning(f"[Jina Smart] Cross-pollination step failed (non-fatal): {e}")


        combined_content: dict[str, str] = {}
        # Label Track A content so AI knows this is the actual company website.
        # The first entry is the homepage; any others are AI-picked subpages.
        for url, text in track_a_result.items():
            label = "COMPANY WEBSITE - subpage picked by link discovery" if url in track_a_picked_urls else "COMPANY WEBSITE - homepage"
            combined_content[url] = f"[SOURCE: {label}]\n{text}"
        # Label Track B content so AI knows these are external search results
        for url, text in track_b_result.items():
            combined_content[url] = f"[SOURCE: WEB SEARCH RESULT for query: '{search_query}']\n{text}"

        if not combined_content:
            # Both tracks empty — don't throw a 500. Return a clean NOTFOUND row
            # so Clay never sees an error. The retry loop treats NOTFOUND as
            # retryable, so this still gets a second attempt before giving up.
            logger.warning("[Jina Smart] Both tracks returned no content — returning NOTFOUND")
            return {
                "track_a_urls": [],
                "track_a_content": {},
                "track_a_picked_subpages": [],
                "track_b_search_query": search_query,
                "track_b_search_results": raw_search_results,
                "track_b_urls": [],
                "track_b_content": {},
                "pages_scraped": 0,
                "total_content_length": 0,
                "extracted_answer": "NOTFOUND",
                "total_tokens": 0,
            }

        logger.info(
            f"[Jina Smart] Combined: {len(track_a_result)} direct + "
            f"{len(track_b_result)} search = {len(combined_content)} pages"
        )

        # ── Final AI extraction ────────────────────────────────────────────────
        extract_response = await ai_client.extract_answer(combined_content, prompt_extract)

        parsed_answer: Any = extract_response.content
        if extract_response.content.strip().upper() != "NOTFOUND":
            try:
                parsed_answer = json.loads(strip_json(extract_response.content))
            except (json.JSONDecodeError, ValueError):
                pass

        return {
            # ── Track A: Jina Reader on the website directly ──────────────────
            "track_a_urls": list(track_a_result.keys()),
            "track_a_content": track_a_result,  # {url: full page text from Jina Reader}
            "track_a_picked_subpages": track_a_picked_urls,  # AI-picked subpages beyond the homepage

            # ── Track B: AI query → Jina Search → snippets to LLM ────────────
            "track_b_search_query": search_query,                # query the AI generated
            "track_b_search_results": raw_search_results,        # raw [{url, title, snippet}] from s.jina.ai
            "track_b_urls": list(track_b_result.keys()),
            "track_b_content": track_b_result,  # {url: "Title + Snippet" sent to LLM}

            # ── Combined ──────────────────────────────────────────────────────
            "pages_scraped": len(combined_content),
            "total_content_length": sum(len(v) for v in combined_content.values()),
            "extracted_answer": parsed_answer,
            "total_tokens": extract_response.input_tokens + extract_response.output_tokens,
        }

    # Retry policy: retry on ANY scrape failure, then ALWAYS return a clean 200.
    # A failed scrape is a RESULT ("NOTFOUND"), not an HTTP error — Clay should
    # never see a 500/504. Each failure is retried once; if it still fails, we
    # return {"extracted_answer": "NOTFOUND", "error": <reason>} as a 200.
    #   - NOTFOUND     → retry, then return NOTFOUND
    #   - Timeout      → retry, then return NOTFOUND
    #   - Both tracks  → retry, then return NOTFOUND
    #   - Exception    → retry, then return NOTFOUND
    #   - 422 (no prompt) → raise — the ONE real error: can't scrape with no prompt
    # Outer hard cap per attempt: 30s (Track A 18s + Track B 12s run parallel + extract ~8s)
    # Typical success ~20-25s; worst case (full fail, 2 attempts) ~61s.
    # Per-attempt budgets. Attempt 1 is FAST (most rows answer quick). The retry
    # is GENEROUS — full time budget, like a fresh manual rerun — because by the
    # time we're retrying, a slow/complete answer beats a fast NOTFOUND.
    #   (track_a_deadline, track_b_deadline, outer_timeout, max_search_queries)
    # The retry uses FEWER search queries (1 vs 2) — by the time we retry, Jina
    # Search is likely rate-limited, so we lighten the load instead of hammering.
    ATTEMPT_BUDGETS = [
        (18.0, 12.0, 35.0, 2),   # attempt 1 — fast path, 2 queries
        (45.0, 30.0, 85.0, 1),   # attempt 2 (retry) — generous time, only 1 query
    ]
    MAX_RETRIES = len(ATTEMPT_BUDGETS)  # 2 full pipeline runs = 1 retry
    RETRY_BACKOFF = 4.0  # gap before the full re-run — long enough for a rate
    #                      limit / Jina blip to actually clear (1s was too short)
    last_result: dict | None = None
    last_error: str | None = None

    def _notfound(error: str | None = None) -> dict:
        """A clean 200 result for any scrape failure — never throw to Clay."""
        return {
            "track_a_urls": [],
            "track_a_content": {},
            "track_a_picked_subpages": [],
            "track_b_search_query": "",
            "track_b_search_results": [],
            "track_b_urls": [],
            "track_b_content": {},
            "pages_scraped": 0,
            "total_content_length": 0,
            "extracted_answer": "NOTFOUND",
            "total_tokens": 0,
            "error": error,
        }

    for attempt in range(1, MAX_RETRIES + 1):
        track_a_deadline, track_b_deadline, per_attempt_timeout, max_queries = ATTEMPT_BUDGETS[attempt - 1]
        try:
            result = await asyncio.wait_for(
                _run(track_a_deadline, track_b_deadline, max_queries),
                timeout=per_attempt_timeout,
            )
            answer = result.get("extracted_answer", "NOTFOUND")

            if isinstance(answer, str) and answer.strip().upper() == "NOTFOUND":
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"[Jina Smart] Attempt {attempt}/{MAX_RETRIES} returned NOTFOUND — retrying..."
                    )
                    last_result = result
                    await asyncio.sleep(RETRY_BACKOFF)
                    continue
                else:
                    logger.warning(
                        f"[Jina Smart] All {MAX_RETRIES} attempts returned NOTFOUND — stopping."
                    )
                    return result

            return result

        except asyncio.TimeoutError:
            last_error = f"Pipeline timed out after {per_attempt_timeout}s"
            logger.error(f"[Jina Smart] Outer timeout on attempt {attempt}/{MAX_RETRIES} ({per_attempt_timeout}s budget)")
            if attempt < MAX_RETRIES:
                logger.warning(f"[Jina Smart] Retrying after timeout...")
                await asyncio.sleep(RETRY_BACKOFF)
                continue
            logger.warning(f"[Jina Smart] All attempts timed out — returning NOTFOUND")
            return _notfound(last_error)
        except HTTPException as he:
            # 422 = genuine client error (missing prompt_extract) — can't scrape
            # without a prompt, so this is the ONE case we still surface as an error.
            if he.status_code == 422:
                raise
            last_error = str(he.detail)
            logger.error(f"[Jina Smart] Attempt {attempt}/{MAX_RETRIES} failed: {he.detail}")
            if attempt < MAX_RETRIES:
                logger.warning(f"[Jina Smart] Retrying after failure...")
                await asyncio.sleep(RETRY_BACKOFF)
                continue
            logger.warning(f"[Jina Smart] All attempts failed — returning NOTFOUND")
            return _notfound(last_error)
        except Exception as e:
            last_error = str(e)
            logger.error(f"[Jina Smart] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.warning(f"[Jina Smart] Retrying after exception...")
                await asyncio.sleep(RETRY_BACKOFF)
                continue
            logger.warning(f"[Jina Smart] All attempts crashed — returning NOTFOUND")
            return _notfound(last_error or str(e))

    # Should never reach here, but satisfy type checker
    if last_result is not None:
        return last_result
    raise HTTPException(status_code=500, detail="Unexpected error in retry loop")


@app.post("/scrape/jina-agent")
async def scrape_jina_agent(request: JinaSmartRequest):
    """
    Agentic version: a LangGraph ReAct agent (Gemini) decides what to read and
    search, using Jina as tools. More flexible navigation than /scrape/jina-test,
    at higher cost/latency. Same request shape. Never throws on a scrape failure
    — returns extracted_answer = "NOTFOUND" on any error (like the pipeline).
    """
    request.normalize()
    prompt_extract = request.prompt_extract or request.data.get("prompt_extract", "")
    if not prompt_extract:
        raise HTTPException(status_code=422, detail="prompt_extract is required (top-level or inside data)")

    PROMPT_KEYS = {"prompt_extract", "prompt_filter"}
    clean_data = {k: v for k, v in request.data.items() if k not in PROMPT_KEYS}

    # Reuse the same robust website detection as the pipeline
    WEBSITE_KEYS = (
        "website", "url", "domain", "website_url", "site", "homepage",
        "company_website", "web", "company_url", "company_domain",
    )
    website_url = ""
    for key in WEBSITE_KEYS:
        if clean_data.get(key):
            website_url = clean_data[key].strip()
            break
    if not website_url:
        for v in clean_data.values():
            s = str(v).strip()
            if (s.startswith(("http://", "https://")) or re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", s, re.I)) \
                    and "linkedin.com" not in s.lower():
                website_url = s
                break
    if website_url and not website_url.startswith(("http://", "https://")):
        website_url = f"https://{website_url}"

    # Provider is resolved inside the agent (aliases, "mix" round-robin, etc.)
    provider = request.ai_provider or "gemini"
    logger.info(f"[Jina Agent] Starting agent (provider={provider}) keys={list(clean_data.keys())} website={website_url}")

    from ai.jina_agent import run_jina_agent

    AGENT_TIMEOUT = float(get_settings().agent_timeout_seconds)  # hard outer cap per row
    try:
        result = await asyncio.wait_for(
            run_jina_agent(clean_data, prompt_extract, website_url, provider=provider),
            timeout=AGENT_TIMEOUT,
        )
        logger.info(f"[Jina Agent] Done in {result.get('tool_calls')} tool calls")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[Jina Agent] Hit {AGENT_TIMEOUT}s timeout — returning NOTFOUND")
        return {"extracted_answer": "NOTFOUND", "provider": provider, "tool_calls": 0,
                "forced_final": False, "error": f"timeout after {AGENT_TIMEOUT}s", "raw_text": None}
    except Exception as e:
        logger.error(f"[Jina Agent] Failed: {e}")
        return {"extracted_answer": "NOTFOUND", "provider": provider, "tool_calls": 0,
                "forced_final": False, "error": str(e), "raw_text": None}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "name": "AI Web Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "/scrape": "POST - Process batch of scrape jobs from Supabase (uses Playwright + AI filtering)",
            "/scrape/scrapingbee-only": "POST - Process UNPROCESSED rows with ScrapingBee only (no Playwright, fast)",
            "/scrape/fallback": "POST - Re-run FAILED rows with ScrapingBee main-page mode",
            "/scrape/single": "POST - Process single scrape request (domain-based)",
            "/scrape/direct": "POST - Process single URL directly without database (standalone scraping)",
            "/health": "GET - Health check",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
