# Jina Test Endpoint (Non-Technical Guide)

## What this endpoint is for

The **Jina Test endpoint** is used to quickly gather reliable public information about a company and return a structured answer to a specific question.

It is designed for enrichment workflows (for example: sales research, lead qualification, or company profiling).

---

## Endpoint name

**`/scrape/jina-test`**

---

## What you provide

You send:

- Basic company details (such as name, website, domain, LinkedIn, etc.)
- One clear extraction instruction (the question you want answered)

---

## How it works (simple view)

The endpoint runs **two information tracks at the same time**:

1. **Direct Website Track**
   - Reads the company website directly.

2. **Web Search Track**
   - Generates a smart search query from the company details.
   - Pulls top web search results and uses their summaries.

Then it combines both tracks and produces one final answer.

---

## Why this is useful

- **Higher reliability:** if one source is weak, the other may still provide value.
- **Faster decision support:** both tracks run in parallel.
- **Better coverage:** includes both first-party (company site) and third-party web context.

---

## What you get back

A response that includes:

- Website source used
- Search query used
- Search results considered
- Final extracted answer
- Basic processing metrics (pages/content/tokens)

If the system can’t find enough evidence, it may return **`NOTFOUND`**.

---

## Reliability and safeguards

- Up to **3 attempts** are made when results are uncertain.
- Each attempt has a strict timeout to avoid long waits.
- If both tracks fail, the endpoint returns a clear error message.

---

## Best practices (business users)

- Write one focused extraction question.
- Provide a valid company website whenever possible.
- Avoid very broad requests; specific prompts produce better outcomes.

---

## Typical business use cases

- Identify a company’s services and target market
- Extract hiring or growth signals
- Summarize positioning or value proposition
- Collect high-level competitor intelligence
