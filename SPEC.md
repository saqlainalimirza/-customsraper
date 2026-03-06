# Scaletopia Custom Scraper — System Spec
_Last updated: March 6, 2026_

---

## What It Does

A **batch web scraping API** that takes a list of company domains, visits each website, uses AI to find and read the most relevant pages, and extracts structured answers to a custom question — all stored in Supabase.

**In plain English:** you give it 500 company domains and ask *"What are their pricing tiers?"* — it crawls every site, picks the right pages, reads them, and returns structured JSON answers for each company.

---

## How It Works (Step by Step)

```
Input: list of domains in Supabase
          ↓
1. Crawl homepage → collect all internal links (up to 30)
          ↓
2. AI filters links → picks up to 5 most relevant URLs
          ↓
3. Scrape those pages → extract clean text content
          ↓
4. AI reads content → returns structured JSON answer
          ↓
Output: answer stored back in Supabase row
```

If the main pipeline fails (bot-blocked, JS-rendered, etc.), a **fallback** kicks in using ScrapingBee to bypass protections and retry.

---

## Inputs

| What | Where it comes from |
|---|---|
| **Domains to scrape** | Pre-loaded into Supabase (`scrape_jobs` table) |
| **Filter prompt** | Sent via API call — e.g. *"Which URLs might contain pricing info?"* |
| **Extract prompt** | Sent via API call — e.g. *"What are the pricing tiers and their features?"* |
| **Dataset ID** | Groups a batch of domains together |

---

## Outputs (per domain)

Each Supabase row gets updated with:

| Field | Description |
|---|---|
| `extracted_answer` | Structured JSON answer, or `"NOTFOUND"` if info wasn't on the site |
| `all_urls` | Every link found on the homepage |
| `filtered_urls` | The ≤5 URLs the AI selected to scrape |
| `scraped_content` | Raw text scraped from those pages |
| `status` | `completed`, `failed`, or intermediate states |
| `total_tokens` | Token count for cost tracking |

---

## API Endpoints

### `POST /scrape` — Main Batch Job
Processes all pending domains in a dataset.

```json
{
  "dataset_id": "my-campaign",
  "prompt_filter": "Which URLs might contain info about pricing or plans?",
  "prompt_extract": "What are the pricing tiers and what features does each include?",
  "limit": 100,
  "ai_provider": "gpt",
  "run_fallback": true
}
```

**Response:**
```json
{
  "processed": 100,
  "successful": 87,
  "failed": 13,
  "total_tokens": 482000,
  "fallback_successful": 9,
  "fallback_failed": 4
}
```

### `POST /scrape/single` — Test a Single Domain
Same pipeline on one domain — good for testing prompts before a full run.

### `POST /scrape/direct` — Scrape a Specific URL
Skips crawling; scrapes one known URL directly.

### `POST /scrape/fallback` — Retry Failed Rows
Re-runs only the rows that previously failed, using ScrapingBee.

### `GET /health` — Health check

---

## Infrastructure

| Component | Tech |
|---|---|
| **API** | FastAPI (Python), runs on port 8000 |
| **Database** | Supabase (Postgres) |
| **AI Models** | GPT-4o-mini or Claude 3.5 Sonnet via OpenRouter |
| **Web Scraping** | HTTP requests + BeautifulSoup (primary) |
| **Fallback Scraping** | ScrapingBee (handles JS-heavy / bot-protected sites) |
| **Deployment** | Docker / Railway |
| **Concurrency** | 30 parallel workers; 3-minute timeout per domain |

---

## Database Table: `scrape_jobs`

You load domains into this table before triggering a run. The API reads `pending` rows and writes results back.

```
dataset_id   → groups domains into a campaign/batch
domain       → the website to scrape (e.g. "acme.com")
processed    → false until completed or failed
status       → pending → crawling → filtering → scraping → extracting → completed / failed
extracted_answer → the final AI output
```

---

## Key Behaviors

- **Prompts are not stored** in the database — they are passed fresh with every API call, so you can rerun the same domains with different questions.
- **`NOTFOUND`** is a valid result — it means the site was scraped successfully but the requested info wasn't there.
- **Token logging** is tracked per row so cost per run can be calculated.
- **Fallback is automatic** — failed rows are retried with ScrapingBee by default unless `run_fallback: false`.
- **Rate limiting** — 2-second delay between batches; random user-agent rotation to reduce blocking.

---

## Typical Usage Flow

1. **Load domains** into Supabase with your `dataset_id`
2. **Call `/scrape`** with your two prompts and dataset ID
3. **Monitor** — rows update in real time in Supabase as they're processed
4. **Query results** — filter by `dataset_id` and `status = 'completed'` to pull answers
