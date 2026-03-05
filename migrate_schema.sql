-- Run this in Supabase SQL Editor to update the schema
-- This removes the prompt columns (prompts now come from API)

-- Option 1: If you want to keep existing data, alter the table
ALTER TABLE scrape_jobs 
DROP COLUMN IF EXISTS prompt_filter,
DROP COLUMN IF EXISTS prompt_extract;

-- Option 2: If you want to start fresh, drop and recreate
-- DROP TABLE IF EXISTS scrape_jobs;
-- Then run schema.sql
