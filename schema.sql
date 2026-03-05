-- Supabase Table Schema for AI Web Scraper
-- Run this in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS scrape_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    
    -- Results
    all_urls JSONB,                     -- All discovered URLs
    filtered_urls JSONB,                -- AI-selected URLs
    scraped_content JSONB,              -- Raw scraped content
    extracted_answer TEXT,              -- Final AI answer
    
    -- Token logging
    filter_input_tokens INTEGER,
    filter_output_tokens INTEGER,
    extract_input_tokens INTEGER,
    extract_output_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Metadata
    status TEXT DEFAULT 'pending',      -- pending, crawling, filtering, scraping, extracting, fallback_scraping, fallback_extracting, completed, failed
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficient querying of unprocessed rows by dataset
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_dataset_processed 
ON scrape_jobs(dataset_id, processed);

-- Index for status queries
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_status 
ON scrape_jobs(status);

-- Example: Insert test rows (prompts are provided via API, not stored here)
-- INSERT INTO scrape_jobs (dataset_id, domain)
-- VALUES 
--     ('test-dataset-1', 'unitzero.tech'),
--     ('test-dataset-1', 'example.com'),
--     ('test-dataset-1', 'another-site.io');
