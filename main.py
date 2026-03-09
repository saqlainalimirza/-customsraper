import json
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
from scraper.jina_scraper import JinaScraper
from ai.openrouter_client import OpenRouterClient
from ai.base import AIClient
from utils.logging import setup_logger, log_pipeline_step, log_summary

logger = setup_logger(__name__)

PARALLEL_WORKERS = 30
# No browser semaphore needed - using simple HTTP now!
FALLBACK_WORKERS = 10
ROW_TIMEOUT = 180
BATCH_DELAY = 2.0  # Delay between batches to prevent rate limits


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Web Scraper API")
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
    data: dict[str, str]  # e.g. {"name": "Acme", "website": "acme.com", "linkedin": "..."}
    # prompt_extract and prompt_filter can be top-level fields OR nested inside data — both work
    prompt_extract: str | None = None
    prompt_filter: str | None = None
    ai_provider: Literal["gpt", "claude"] = "gpt"


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
    if provider in ("gpt", "claude"):
        return OpenRouterClient(model_type=provider)
    else:
        raise ValueError(f"Unknown AI provider: {provider}. Use 'gpt' or 'claude'")


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
                parsed_answer = json.loads(extracted_answer)
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
    Hard timeout: 55s total.
    """
    logger.info(f"[Jina Smart] Starting pipeline with data keys: {list(request.data.keys())}")

    async def _run() -> dict:
        ai_client = get_ai_client(request.ai_provider)
        jina = JinaScraper()

        # Allow prompts nested inside data (Clay/Zapier style)
        prompt_extract = request.prompt_extract or request.data.get("prompt_extract", "")
        if not prompt_extract:
            raise HTTPException(status_code=422, detail="prompt_extract is required (top-level or inside data)")

        # Strip prompt keys — only company fields go to AI for query generation
        PROMPT_KEYS = {"prompt_extract", "prompt_filter"}
        clean_data = {k: v for k, v in request.data.items() if k not in PROMPT_KEYS}

        website_url = (
            clean_data.get("website")
            or clean_data.get("url")
            or clean_data.get("domain")
            or ""
        ).strip()

        # ── TRACK A: Direct website scrape ────────────────────────────────────
        async def run_track_a() -> dict[str, str]:
            if not website_url:
                logger.warning("[Jina Smart][Track A] No website provided, skipping")
                return {}
            try:
                url, content = await jina.scrape_main_page(website_url)
                logger.info(f"[Jina Smart][Track A] {len(content)} chars from {url}")
                return {url: content}
            except Exception as e:
                logger.warning(f"[Jina Smart][Track A] Failed for {website_url}: {e}")
                return {}

        # ── TRACK B: Search → parallel read ───────────────────────────────────
        async def run_track_b() -> tuple[str, dict[str, str]]:
            query_response = await ai_client.generate_search_query(clean_data, prompt_extract)
            search_query = query_response.content.strip()
            logger.info(f"[Jina Smart][Track B] Search query: '{search_query}'")
            if not search_query:
                return "", {}

            search_results = await jina.search(search_query)
            if not search_results:
                logger.warning(f"[Jina Smart][Track B] No results for: '{search_query}'")
                return search_query, {}

            # Cap at 3, skip same domain as Track A
            filtered = [
                r for r in search_results
                if not (website_url and extract_domain(r["url"]) == extract_domain(website_url))
            ][:3]

            async def read_one(result: dict) -> tuple[str, str]:
                url = result["url"]
                try:
                    content = await jina.scrape_url(url)
                    logger.info(f"[Jina Smart][Track B] {len(content)} chars from {url}")
                    return url, content
                except Exception as e:
                    logger.warning(f"[Jina Smart][Track B] Reader failed for {url}: {e}")
                    snippet = result.get("content", "")
                    return url, snippet

            pairs = await asyncio.gather(*[read_one(r) for r in filtered])
            content_map = {url: content for url, content in pairs if content}
            return search_query, content_map

        # ── Both tracks in parallel ────────────────────────────────────────────
        track_a_result, (search_query, track_b_result) = await asyncio.gather(
            run_track_a(),
            run_track_b(),
        )

        combined_content: dict[str, str] = {}
        combined_content.update(track_a_result)
        combined_content.update(track_b_result)

        if not combined_content:
            raise HTTPException(status_code=500, detail="Both tracks failed — no content scraped")

        logger.info(
            f"[Jina Smart] Combined: {len(track_a_result)} direct + "
            f"{len(track_b_result)} search = {len(combined_content)} pages"
        )

        # ── Final AI extraction ────────────────────────────────────────────────
        extract_response = await ai_client.extract_answer(combined_content, prompt_extract)

        parsed_answer: Any = extract_response.content
        if extract_response.content.strip().upper() != "NOTFOUND":
            try:
                parsed_answer = json.loads(extract_response.content)
            except (json.JSONDecodeError, ValueError):
                pass

        return {
            "track_a_urls": list(track_a_result.keys()),
            "track_b_search_query": search_query,
            "track_b_urls": list(track_b_result.keys()),
            "pages_scraped": len(combined_content),
            "total_content_length": sum(len(v) for v in combined_content.values()),
            "scraped_content": combined_content,
            "extracted_answer": parsed_answer,
            "total_tokens": extract_response.input_tokens + extract_response.output_tokens,
        }

    try:
        return await asyncio.wait_for(_run(), timeout=55.0)
    except asyncio.TimeoutError:
        logger.error("[Jina Smart] Pipeline timed out after 55s")
        raise HTTPException(status_code=504, detail="Pipeline timed out — site may be slow or blocking requests")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Jina Smart] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
