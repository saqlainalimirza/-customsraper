# AI Web Scraper API

A FastAPI-based web scraper that crawls domains, uses AI to filter relevant URLs, scrapes content, and extracts answers - integrated with Supabase for batch processing.

## Features

- **Domain Crawling**: Discovers all URLs on a domain using Crawl4AI with anti-blocking measures
- **AI URL Filtering**: Uses OpenAI or Anthropic to identify URLs likely to contain relevant information
- **Content Scraping**: Extracts clean text content from filtered URLs
- **AI Answer Extraction**: Analyzes scraped content to answer your specific questions
- **Automatic Fallback Pipeline**: Re-processes failed rows with ScrapingBee by scraping only the main website page and running extract prompt directly
- **Token Logging**: Full logging of input/output tokens for cost tracking
- **Supabase Integration**: Batch processing with persistent job tracking

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
crawl4ai-setup  # Required for Crawl4AI browser setup
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Set Up Supabase Table

Run the SQL in `schema.sql` in your Supabase SQL Editor.

### 4. Run the API

```bash
python main.py
# Or with uvicorn:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run with Docker

```bash
docker build -t ai-web-scraper .
docker run --rm -p 8000:8000 --env-file .env ai-web-scraper
```

### 6. Run with Docker Compose

```bash
docker compose up --build
```

API is available at `http://localhost:8000`.

### 7. Deploy on Railway

- Push this project to GitHub
- In Railway, create a new project from the GitHub repo
- Railway will detect the `Dockerfile` and deploy it
- Add all env vars from `.env.example` in Railway Variables
- Railway uses the `PORT` environment variable automatically (already supported in `Dockerfile`)

## API Endpoints

### POST /scrape

Process a batch of scrape jobs from Supabase. **Prompts are provided via API, not stored in database.**

```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my-dataset",
    "prompt_filter": "Which URLs might contain information about pricing or plans?",
    "prompt_extract": "What are the pricing tiers and what features does each include?",
    "limit": 100,
    "ai_provider": "gpt",
    "run_fallback": true
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_id` | string | Yes | ID to fetch unprocessed rows |
| `prompt_filter` | string | Yes | Prompt 1: Which URLs might contain info about X? |
| `prompt_extract` | string | Yes | Prompt 2: Based on scraped content, answer X? |
| `limit` | int | No | Max rows to process (default: 100) |
| `ai_provider` | string | No | "gpt" or "claude" (default: "gpt") |
| `run_fallback` | bool | No | If true, runs ScrapingBee fallback on failed rows (default: true) |
| `fallback_limit` | int | No | Max failed rows to retry in fallback (default: same as `limit`) |

**Response:**
```json
{
  "processed": 10,
  "successful": 8,
  "failed": 2,
  "total_tokens": 17420,
  "fallback_processed": 2,
  "fallback_successful": 1,
  "fallback_failed": 1
}
```

### POST /scrape/fallback

Run fallback-only mode for failed rows. This pipeline skips URL discovery and URL filtering, scrapes only the website main page via ScrapingBee, then runs `prompt_extract`.

```bash
curl -X POST http://localhost:8000/scrape/fallback \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "my-dataset",
    "prompt_extract": "What services does this company provide?",
    "limit": 100,
    "ai_provider": "gpt"
  }'
```

### POST /scrape/single

Process a single scrape request without Supabase (for testing).

```bash
curl -X POST http://localhost:8000/scrape/single \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "unitzero.tech",
    "prompt_filter": "Which URLs might contain information about the team?",
    "prompt_extract": "Who are the founders and what are their backgrounds?",
    "ai_provider": "openai"
  }'
```

## Supabase Table Schema

The table only stores domains and results. **Prompts are provided via API request.**

```sql
CREATE TABLE scrape_jobs (
    id UUID PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,              -- Only domain, no prompts stored
    processed BOOLEAN DEFAULT FALSE,
    
    -- Results
    all_urls JSONB,
    filtered_urls JSONB,
    scraped_content JSONB,
    extracted_answer TEXT,
    
    -- Token logging
    filter_input_tokens INTEGER,
    filter_output_tokens INTEGER,
    extract_input_tokens INTEGER,
    extract_output_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Metadata
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);
```

## How It Works

1. **Insert rows** into `scrape_jobs` with `dataset_id` and `domain` only
2. **Call the API** with `dataset_id`, `limit`, and your **two prompts**
3. **For each row**, the pipeline:
   - Crawls the domain to discover all URLs
   - AI filters URLs based on `prompt_filter` (from API)
   - Scrapes content from filtered URLs
   - AI extracts answer based on `prompt_extract` (from API)
4. **Fallback stage (optional)**:
   - Fetches rows with `status='failed'` from the same dataset
   - Uses ScrapingBee to scrape only each site main page (no link crawling/filtering)
   - Runs AI extraction using only `prompt_extract`
5. **Results** are saved back to Supabase with token counts

## Anti-Blocking Features

- Realistic browser headers (Sec-Fetch-*, Accept, etc.)
- User-Agent rotation (10+ recent Chrome/Firefox agents)
- Random delays between requests (1-3 seconds)
- Headless browser with stealth mode
- Session persistence and cookie handling

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_KEY` | Supabase anon key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model | gpt-4o-mini |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `ANTHROPIC_MODEL` | Anthropic model | claude-3-5-sonnet-20241022 |
| `MAX_URLS_PER_DOMAIN` | Max URLs to crawl | 500 |
| `REQUEST_DELAY_MIN` | Min delay (seconds) | 1.0 |
| `REQUEST_DELAY_MAX` | Max delay (seconds) | 3.0 |
| `SCRAPINGBEE_API_KEY` | ScrapingBee API key (fallback pipeline) | Optional unless fallback is used |
| `SCRAPINGBEE_TIMEOUT_SECONDS` | ScrapingBee request timeout | 45 |

## Project Structure

```
scaletopia-custom-scrapper/
├── main.py                 # FastAPI app + endpoints
├── config.py               # Settings management
├── schema.sql              # Supabase table schema
├── scraper/
│   ├── crawler.py          # Domain URL discovery
│   └── content.py          # Content scraping
├── ai/
│   ├── base.py             # AI client interface
│   ├── openai_client.py    # OpenAI implementation
│   ├── anthropic_client.py # Anthropic implementation
│   └── prompts.py          # Prompt templates
├── db/
│   └── supabase_client.py  # Supabase operations
└── utils/
    └── logging.py          # Structured logging
```
