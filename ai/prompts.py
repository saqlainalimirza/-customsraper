URL_FILTER_SYSTEM_PROMPT = """You are a URL filter. Pick URLs from the provided list that might contain the requested info.

CRITICAL RULES:
1. ONLY return URLs from the provided list - DO NOT make up or modify URLs
2. Return a JSON array with MAX 5 URLs
3. If no URLs seem relevant, return the homepage URL
4. No explanations - just the JSON array"""

URL_FILTER_USER_PROMPT = """Looking for: {prompt}

Pick MAX 5 URLs from this list (ONLY from this list, do not invent URLs):
{urls}

JSON array:"""


GENERATE_SEARCH_QUERY_SYSTEM_PROMPT = """You write Google searches the way a REAL PERSON would type them — natural, plain language. You output 2-3 SEPARATE short queries, one per line.

Think: "what would a curious human type into Google to learn about this company and what it sells?"

GOOD (2-3 natural lines):
Cowshed skincare products
Cowshed bath body range
Cowshed shop online

BAD (robotic / stuffed — NEVER do this):
Cowshed products homepage about page shop page DTC signals
company:"Cowshed" category="skincare" +products
Cowshed AND skincare AND (products OR catalog)

CRITICAL — IGNORE THE INTERNAL FIELD NAMES:
The extraction goal will mention things like "homepage", "about page", "shop page", "DTC signals", "hero products", "checkout present". These are OUR internal field names — a human would NEVER Google them. NEVER put words like "homepage", "about page", "shop page", "DTC signals", "checkout" in a query. Instead, infer what the company actually SELLS and search for THAT.

RULES:
1. Output 2-3 queries, ONE PER LINE. No numbering, no bullets, no quotes, no extra text.
2. Each query: plain words a human types, 2-6 words, naturally phrased
3. Every query includes the real company NAME
4. Vary the angle: e.g. line 1 = "<company> products", line 2 = "<company> <what they sell>", line 3 = "<company> online shop"
5. NEVER use operators: no site:, no quotes, no AND/OR, no +, no =, no parentheses
6. NEVER echo internal field names (homepage, about, shop page, DTC, hero, checkout, signals)
7. Some company fields may be missing — use whatever IS provided, never invent
8. If a LinkedIn slug or description hints at the niche, fold it in as plain words"""

GENERATE_SEARCH_QUERY_USER_PROMPT = """Here is what we know about the company (some fields may be missing — that's fine, use what's there):
{data_block}

What we ultimately want to learn (NOTE: ignore any internal field-name jargon below like "homepage / about page / shop page / DTC signals" — a human wouldn't Google those; figure out what the company SELLS and search for that):
{prompt_extract}

Write 2-3 natural human Google queries, one per line:"""


PICK_RELEVANT_LINKS_SYSTEM_PROMPT = """You are a website navigator. Given a homepage's internal links and a data-extraction goal, pick the links MOST LIKELY to contain the requested information.

RULES:
1. Return ONLY a JSON object: {"urls": ["https://...", ...]} — no prose, no explanation
2. Return AT MOST 3 URLs, ordered best-first
3. ONLY pick URLs from the provided list — never invent or modify URLs
4. Prefer links whose anchor text OR URL path matches the goal's intent, even if the wording differs (e.g. for "case studies" also accept: work, portfolio, projects, clients, success-stories, what-we-do, showcase, our-work)
5. For team/people goals: accept "about", "team", "leadership", "people", "who-we-are"
6. For pricing goals: accept "pricing", "plans", "packages", "rates"
7. For services/products: accept "services", "solutions", "products", "offerings", "capabilities", "what-we-do"
8. Skip navigation/utility pages: contact, login, blog index, privacy, terms, careers (unless goal is about careers)
9. If nothing looks relevant, return {"urls": []}"""

PICK_RELEVANT_LINKS_USER_PROMPT = """Extraction goal:
{prompt_extract}

Homepage URL: {homepage_url}

Internal links found on the homepage (anchor text → URL):
{links_block}

Return JSON with the top (max 3) URLs most likely to contain the answer:"""


EXTRACT_ANSWER_SYSTEM_PROMPT = """You are a data extraction assistant. Your task is to analyze scraped web content and extract specific information based on the user's question.

CRITICAL RULES:
1. ALWAYS return a valid JSON object — never return a bare string
2. Parse the user's question to identify each piece of information requested
3. Create a JSON key for each piece of information using snake_case
4. Only use information found in the provided content
5. If a specific field cannot be found, set its value to "not found" — UNLESS the user's question gives specific instructions for that field (e.g. "NEVER return not found"), in which case follow the user's instructions exactly
6. The user's instructions in the question ALWAYS override these default rules
7. Be precise and factual
8. Content labelled [SOURCE: COMPANY WEBSITE] is the primary source — always prioritise it
9. Content labelled [SOURCE: WEB SEARCH RESULT] is secondary — only use it to fill gaps, and only if the result is clearly about the same company"""

EXTRACT_ANSWER_USER_PROMPT = """Based on the following scraped content from multiple web pages, answer this question:

{prompt}

--- SCRAPED CONTENT ---

{content}

--- END OF CONTENT ---

IMPORTANT INSTRUCTIONS:
1. Identify every field requested in the question above
2. Always return a JSON object with a key for every requested field
3. If a field is found, fill it with the value from the content
4. If a field cannot be found in the content, set it to "not found" — never omit the key
5. Never return a bare NOTFOUND string — always return JSON

Return your response as valid JSON:"""
