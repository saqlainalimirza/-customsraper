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
# page dumps blow up tokens-per-minute (TPM). Trim hard.
TOOL_OUTPUT_CHARS = 6000

# Cap concurrent agent runs so parallel rows don't burst past the Gemini TPM
# limit all at once. Module-level → shared across all requests. Kept low (2)
# because each agent re-sends its growing context every step — parallel agents
# stack tokens-per-minute fast and trip the 1M TPM ceiling.
_AGENT_SEMAPHORE = asyncio.Semaphore(2)


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


@tool
async def search_web(query: str) -> str:
    """Search the web (Google-style) and return the top results as JSON with
    url, title, and snippet for each. Use this when the company website does
    not have the info, or to discover the right pages to read_url next.
    Write a natural human query like 'Acme skincare products'."""
    jina = JinaScraper()
    try:
        results = await jina.search(query)
        logger.info(f"[Jina Agent] search_web('{query}') → {len(results)} results")
        if not results:
            return "No results found. Try a different, simpler query."
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

STRATEGY:
1. Start by reading the company homepage.
2. Look at the links/menu in the homepage text and read the pages most likely to hold the answer (shop, products, collections, about, pricing, etc.).
3. If the site doesn't have what you need, use search_web with a natural human query, then read_url the best result.
4. Be efficient — read AT MOST 4-5 pages total. You have a LIMITED number of steps.

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


def _build_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        # NO retries on the agent — call Gemini once, take the answer. The agent
        # endpoint exists precisely to avoid the retry path (that's only on
        # /scrape/jina-test). langchain defaults to max_retries=6; force it to 0.
        max_retries=0,
        # LOW thinking — faster, cheaper, fewer thinking-tokens. Gemini 3 uses
        # thinking_level; passed through generation config.
        model_kwargs={"generation_config": {"thinking_config": {"thinking_level": "low"}}},
    )


def _build_agent():
    return create_react_agent(_build_llm(), tools=[read_url, search_web])


async def _force_final_answer(prompt_extract: str, messages: list) -> str:
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

    combined = "\n\n---\n\n".join(gathered)[:60000]
    llm = _build_llm()
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
    max_steps: int = 16,  # langgraph counts EVERY node (LLM + tool) as a step.
    #                       ~16 ≈ 7 tool cycles + final answer. Lower = the agent
    #                       gets cut off mid-research ("need more steps").
) -> dict:
    """
    Run the ReAct agent for one company row.
    Returns {"extracted_answer": <parsed JSON or str>, "steps": int, "messages": [...]}.
    """
    agent = _build_agent()

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
    if isinstance(final_text, list):  # Gemini can return content as parts
        final_text = "".join(p if isinstance(p, str) else p.get("text", "") for p in final_text)

    # Strip code fences and parse
    text = final_text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    text = text.strip()

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
        text = (await _force_final_answer(prompt_extract, messages)).strip()
        forced = True
        if text.startswith("```"):
            text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:text.rfind("```")]
        text = text.strip()
        if not text:
            return {"extracted_answer": "NOTFOUND", "tool_calls": tool_calls,
                    "error": "agent hit step limit, nothing gathered"}

    parsed = text
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "extracted_answer": parsed,
        "tool_calls": tool_calls,
        "total_messages": len(messages),
        "forced_final": forced,
    }
