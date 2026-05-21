"""
LangGraph ReAct agent that researches a company using Jina as tools.

Unlike the fixed two-track pipeline (/scrape/jina-test), here Gemini decides
what to do: it reads the homepage, navigates to the pages it thinks matter,
and searches the web when the site doesn't have the answer. It runs a bounded
tool-calling loop and returns the final JSON.

Tools given to the model:
  - read_url(url)      → Jina Reader (r.jina.ai), clean page text
  - search_web(query)  → Jina Search (s.jina.ai), top results

Bounded by recursion_limit (max tool steps) + an outer asyncio timeout in the
endpoint, so a single row can never run away.
"""
import json
import asyncio

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from scraper.jina_scraper import JinaScraper
from utils.logging import setup_logger

logger = setup_logger(__name__)

# ReAct keeps every tool result in context and re-sends it each step, so big
# page dumps blow up tokens-per-minute (TPM) AND memory. Trim hard.
TOOL_OUTPUT_CHARS = 4000

# Cap concurrent agent runs (module-level → shared across all requests). Must be
# high enough to match Clay's burst, or queued rows time out WAITING for a slot.
# Tune via AGENT_CONCURRENCY env. Lower only if Gemini TPM becomes a problem.
_AGENT_SEMAPHORE = asyncio.Semaphore(get_settings().agent_concurrency)


@tool
async def read_url(url: str) -> str:
    """Read a single web page and return its clean text content.
    Use this to read the company homepage and any subpage (products, shop,
    collections, about, pricing). Pass a full URL like https://example.com/shop."""
    jina = JinaScraper()
    try:
        content = await jina.scrape_url(url, keep_links=True)
        logger.info(f"[Jina Agent] read_url({url}) → {len(content)} chars (trimmed to {TOOL_OUTPUT_CHARS})")
        return content[:TOOL_OUTPUT_CHARS]
    except Exception as e:
        logger.warning(f"[Jina Agent] read_url({url}) failed: {e}")
        return f"ERROR: could not read {url} ({e}). Try a different URL or use search_web."


# Social/marketplace/aggregator domains that pollute search results — the agent
# wastes a read step on these instead of the brand's real pages. Drop them.
_JUNK_DOMAINS = (
    "instagram.com", "facebook.com", "youtube.com", "tiktok.com", "pinterest.com",
    "reddit.com", "linkedin.com", "twitter.com", "x.com", "amazon.com",
    "etsy.com", "ebay.com", "yelp.com", "wikipedia.org", "crunchbase.com",
)


def _is_junk(url: str) -> bool:
    u = (url or "").lower()
    return any(d in u for d in _JUNK_DOMAINS)


def _clean_query(q: str) -> str:
    """Strip search operators that make Jina Search 422 or return junk:
    double-quotes, AND/OR, parentheses, and site: filters. Plain words search
    far better and never error."""
    import re as _re
    q = q.replace('"', " ").replace("(", " ").replace(")", " ")
    q = _re.sub(r"\bsite:\S+", " ", q, flags=_re.I)   # drop site:domain
    q = _re.sub(r"\b(OR|AND)\b", " ", q)               # drop boolean ops
    q = _re.sub(r"\s+", " ", q).strip()
    return q


@tool
async def search_web(query: str) -> str:
    """Search the web and return the top results as JSON with url, title, and
    snippet for each. Use this when the company website does not have the info,
    or to discover the right pages to read_url next.
    Write a SIMPLE, NATURAL query like 'Acme skincare products' or
    'Acme wholesale'. Do NOT use quotes, site:, OR, AND, or parentheses —
    plain keywords work best and operator queries fail."""
    jina = JinaScraper()
    query = _clean_query(query)
    try:
        results = await jina.search(query)
        # Drop social/marketplace junk so the agent reads real brand pages
        results = [r for r in results if not _is_junk(r.get("url", ""))]
        logger.info(f"[Jina Agent] search_web('{query}') → {len(results)} results (junk filtered)")
        if not results:
            return "No useful results found. Try a different, simpler query."
        return json.dumps([
            {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")}
            for r in results
        ])
    except Exception as e:
        logger.warning(f"[Jina Agent] search_web('{query}') failed: {e}")
        return f"ERROR: search failed ({e}). Try reading the website directly with read_url."


AGENT_SYSTEM_PROMPT = """You are a B2B web research agent. Your job: find the exact information requested about a company and return it as ONE valid JSON object.

You have two tools:
- read_url(url): read a web page's text
- search_web(query): search Google for results (url + snippet)

STRATEGY (be FAST — you have a tight step budget):
1. Read the company homepage first.
2. From the homepage links, read the ONE best products/shop/collections page. That + the homepage usually answers everything — then ANSWER.
3. Only use search_web if the website genuinely lacks the info (e.g. wholesale/B2B). When you do, use a SIMPLE keyword query (e.g. "Acme wholesale") — never quotes/site:/OR.
4. Read 2-3 pages total, then STOP and write the JSON. Do NOT keep searching for every field — fill unknowns with "not found".

CRITICAL — YOU MUST ALWAYS PRODUCE A FINAL ANSWER:
- You have a limited step budget. Do NOT keep researching forever.
- After ~4-5 tool calls, STOP calling tools and write the final JSON immediately.
- It is ALWAYS better to answer with partial data than to run out of steps with no answer.
- For any field you could not find, put "not found" (unless the user's rules say otherwise) — but STILL return the complete JSON object.
- Your VERY LAST message must be the final JSON answer, nothing else.

OUTPUT RULES:
- Respond with ONLY the final JSON object. No prose, no markdown fences.
- Follow the exact field names and rules in the user's request.
- Never invent values — use "not found" for anything you couldn't confirm.
- NEVER end your turn without the JSON. Always answer."""


def _build_llm(provider: str = "gemini"):
    """Build the agent LLM for a provider/alias via the SHARED model registry
    (same routing the pipeline uses). NO retries on any path."""
    from .model_registry import resolve
    cfg = resolve(provider)

    if cfg["route"] == "openrouter":
        from langchain_openai import ChatOpenAI
        extra = {}
        if cfg["extra_body"]:
            extra["model_kwargs"] = {"extra_body": cfg["extra_body"]}
        logger.info(f"[Jina Agent] LLM → {cfg['model']} (via OpenRouter)")
        return ChatOpenAI(
            model=cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
            temperature=0.2, max_retries=0, **extra,
        )

    # native Gemini
    logger.info(f"[Jina Agent] LLM → {cfg['model']} (native Google)")
    return ChatGoogleGenerativeAI(
        model=cfg["model"], google_api_key=cfg["api_key"],
        temperature=0.2, max_retries=0,
        model_kwargs={"generation_config": {"thinking_config": {"thinking_level": "low"}}},
    )


def _build_agent(provider: str = "gemini"):
    return create_react_agent(_build_llm(provider), tools=[read_url, search_web])


def _extract_json(text: str):
    """Robustly pull a JSON object out of ANY model's output — handles code
    fences, leading prose, and trailing junk. Returns a dict, or None if no
    valid JSON is present. Model-agnostic (Gemini/Grok/Flash Lite all differ)."""
    if not text:
        return None
    t = text.strip()
    # strip ``` / ```json fences
    if t.startswith("```"):
        t = t[t.index("\n") + 1:] if "\n" in t else t[3:]
    if t.endswith("```"):
        t = t[:t.rfind("```")]
    t = t.strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else {"value": v}
    except (json.JSONDecodeError, ValueError):
        pass
    # fallback: grab the first {...} block
    import re as _re
    m = _re.search(r"\{.*\}", t, _re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else {"value": v}
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _normalize_result(
    *, raw_text: str, provider: str, tool_calls: int,
    forced_final: bool = False, error: str | None = None,
) -> dict:
    """SINGLE canonical output shape for every model/path. extracted_answer is
    ALWAYS either a parsed JSON dict or the string "NOTFOUND" — never raw model
    prose — so downstream (Clay) gets one predictable structure every time."""
    parsed = _extract_json(raw_text)
    if parsed is None:
        # couldn't parse JSON from the model — treat as not found, keep raw for debugging
        return {
            "extracted_answer": "NOTFOUND",
            "provider": provider,
            "tool_calls": tool_calls,
            "forced_final": forced_final,
            "error": error or "model returned no parseable JSON",
            "raw_text": (raw_text or "")[:2000],
        }
    return {
        "extracted_answer": parsed,
        "provider": provider,
        "tool_calls": tool_calls,
        "forced_final": forced_final,
        "error": error,
        "raw_text": None,
    }


async def _force_final_answer(prompt_extract: str, messages: list, provider: str = "gemini") -> str:
    """
    Last resort: the agent ran out of steps without writing JSON. Take every
    page/snippet it gathered (the ToolMessage contents) and do ONE plain LLM
    call to synthesize the final JSON. Guarantees an answer instead of NOTFOUND.
    """
    gathered = [
        str(m.content) for m in messages
        if m.__class__.__name__ == "ToolMessage" and getattr(m, "content", "")
    ]
    if not gathered:
        return ""  # nothing was scraped — caller falls back to NOTFOUND

    combined = "\n\n---\n\n".join(gathered)[:30000]
    llm = _build_llm(provider)
    resp = await llm.ainvoke([
        SystemMessage(content=(
            "You are a data extractor. Based ONLY on the research text provided, "
            "output the final JSON answer NOW. No prose, no markdown fences. "
            "Use \"not found\" for any field you cannot fill from the text."
        )),
        HumanMessage(content=(
            f"Extraction goal (follow field rules exactly):\n{prompt_extract}\n\n"
            f"Research gathered so far:\n{combined}\n\n"
            f"Return ONLY the final JSON object now."
        )),
    ])
    out = resp.content
    if isinstance(out, list):
        out = "".join(p if isinstance(p, str) else p.get("text", "") for p in out)
    logger.info("[Jina Agent] Forced final answer from gathered research")
    return out or ""


async def run_jina_agent(
    data: dict[str, str],
    prompt_extract: str,
    website_url: str,
    provider: str = "gemini",  # "gemini" (native) or "grok" (OpenRouter)
    max_steps: int = 12,  # langgraph counts EVERY node (LLM + tool) as a step.
    #                       ~12 ≈ 5 tool cycles + final answer. Faster rows = more
    #                       throughput; the forced-final fallback covers cutoffs.
) -> dict:
    """
    Run the ReAct agent for one company row.
    Returns {"extracted_answer": <parsed JSON or str>, "steps": int, "messages": [...]}.
    """
    agent = _build_agent(provider)

    data_block = "\n".join(f"{k}: {v}" for k, v in data.items())
    human = (
        f"Company info we already have:\n{data_block}\n\n"
        f"Website to start from: {website_url or '(none given — use search_web)'}\n\n"
        f"What to extract (follow these field rules exactly):\n{prompt_extract}\n\n"
        f"Research the company and return ONLY the final JSON object."
    )

    # Throttle concurrent agents to stay under the Gemini TPM ceiling
    async with _AGENT_SEMAPHORE:
        result = await agent.ainvoke(
            {"messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=human)]},
            config={"recursion_limit": max_steps},
        )

    messages = result.get("messages", [])
    final_text = messages[-1].content if messages else ""
    if isinstance(final_text, list):  # some models return content as parts
        final_text = "".join(p if isinstance(p, str) else p.get("text", "") for p in final_text)
    text = (final_text or "").strip()

    tool_calls = sum(
        1 for m in messages
        if getattr(m, "tool_calls", None)
        for _ in m.tool_calls
    )

    # If the agent ran out of steps without writing JSON, DON'T give up —
    # synthesize a final answer from everything it already gathered.
    forced = False
    if "need more steps" in text.lower() or not text:
        logger.warning("[Jina Agent] Hit step limit — forcing final answer from gathered research")
        text = (await _force_final_answer(prompt_extract, messages, provider)).strip()
        forced = True

    # SINGLE canonical structure for every model/path.
    return _normalize_result(
        raw_text=text, provider=provider, tool_calls=tool_calls, forced_final=forced,
    )
