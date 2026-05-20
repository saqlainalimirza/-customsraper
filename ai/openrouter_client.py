import json
from openai import AsyncOpenAI

from config import get_settings
from utils.logging import setup_logger, log_tokens
from .base import AIClient, AIResponse
from .prompts import (
    URL_FILTER_SYSTEM_PROMPT,
    URL_FILTER_USER_PROMPT,
    EXTRACT_ANSWER_SYSTEM_PROMPT,
    EXTRACT_ANSWER_USER_PROMPT,
    GENERATE_SEARCH_QUERY_SYSTEM_PROMPT,
    GENERATE_SEARCH_QUERY_USER_PROMPT,
    PICK_RELEVANT_LINKS_SYSTEM_PROMPT,
    PICK_RELEVANT_LINKS_USER_PROMPT,
)

logger = setup_logger(__name__)


class OpenRouterClient(AIClient):
    """
    OpenAI-compatible client.
      - gpt / claude  → OpenRouter
      - gemini        → Google's native Gemini API (OpenAI-compatible endpoint)
                        using the direct GEMINI_API_KEY
    """

    def __init__(self, model_type: str = "gpt"):
        self.settings = get_settings()

        if model_type == "gemini":
            # Direct Google Gemini — NOT via OpenRouter. Uses GEMINI_API_KEY.
            self.client = AsyncOpenAI(
                api_key=self.settings.gemini_api_key,
                base_url=self.settings.gemini_base_url,
            )
            self.model = self.settings.gemini_model
        else:
            self.client = AsyncOpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
            self.model = self.settings.claude_model if model_type == "claude" else self.settings.gpt_model

        self.model_type = model_type
        # Gemini: run with LOW thinking — faster + far fewer thinking-tokens (TPM).
        # Passed via extra_body so it goes straight into the request body that
        # Google's OpenAI-compat endpoint reads.
        self.extra_body = {"reasoning_effort": "low"} if model_type == "gemini" else None

    async def filter_urls(
        self,
        urls: list[str],
        prompt: str,
        domain: str,
    ) -> AIResponse:
        """Filter URLs using OpenRouter (GPT or Claude)."""
        logger.info(f"Filtering {len(urls)} URLs with OpenRouter ({self.model})")
        
        urls_text = "\n".join(urls)
        user_message = URL_FILTER_USER_PROMPT.format(
            prompt=prompt,
            urls=urls_text,
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": URL_FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            extra_headers={
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
            extra_body=self.extra_body,
        )
        
        content = response.choices[0].message.content or "[]"
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        log_tokens(
            logger,
            operation="filter_urls",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "urls" in parsed:
                content = json.dumps(parsed["urls"])
            elif isinstance(parsed, list):
                content = json.dumps(parsed)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                content = match.group(0)
            else:
                logger.warning("Failed to parse URL filter response as JSON")
                content = "[]"
        
        return AIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw_response=response,
        )

    async def extract_answer(
        self,
        scraped_content: dict[str, str],
        prompt: str,
    ) -> AIResponse:
        """Extract answer from scraped content using OpenRouter."""
        logger.info(f"Extracting answer from {len(scraped_content)} pages with OpenRouter ({self.model})")
        
        content_parts = []
        for url, text in scraped_content.items():
            content_parts.append(f"=== URL: {url} ===\n{text}\n")
        
        combined_content = "\n".join(content_parts)
        
        # gpt-4o-mini supports 128k tokens (~500k chars). Old 100k cap was throwing
        # away Track B + subpage content. Bump to give the LLM full evidence.
        max_chars = 180000 if "gpt" in self.model.lower() else 200000
        if len(combined_content) > max_chars:
            logger.warning(f"Content too long ({len(combined_content)} chars), truncating to {max_chars}")
            combined_content = combined_content[:max_chars] + "\n... [truncated]"
        
        user_message = EXTRACT_ANSWER_USER_PROMPT.format(
            prompt=prompt,
            content=combined_content,
        )
        
        # No max_tokens for Gemini — let it use its full output budget. Only cap
        # the OpenRouter models (where a missing cap can truncate big JSON).
        create_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "extra_headers": {
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
        }
        if self.model_type != "gemini":
            create_kwargs["max_tokens"] = 8000
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        response = await self.client.chat.completions.create(**create_kwargs)
        
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        log_tokens(
            logger,
            operation="extract_answer",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        return AIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw_response=response,
        )

    async def generate_search_query(
        self,
        data: dict[str, str],
        prompt_extract: str,
    ) -> AIResponse:
        """
        Generate 2-3 natural human web search queries from input data + goal.
        Returns AIResponse whose content is a JSON array of query strings.
        """
        data_block = "\n".join(f"{k}: {v}" for k, v in data.items())
        user_message = GENERATE_SEARCH_QUERY_USER_PROMPT.format(
            data_block=data_block,
            prompt_extract=prompt_extract,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATE_SEARCH_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
            extra_body=self.extra_body,
        )

        raw = (response.choices[0].message.content or "").strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Parse one-query-per-line → clean list. Strip numbering/bullets/quotes.
        import re as _re
        queries: list[str] = []
        for line in raw.splitlines():
            q = line.strip().strip('"').strip("'")
            q = _re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", q).strip()  # drop "1." / "- " / "* "
            # drop any leftover operators a model might sneak in
            q = q.replace('"', "").replace("site:", "").strip()
            if q and len(q) > 2 and q not in queries:
                queries.append(q)
        queries = queries[:3] or ([raw] if raw else [])

        log_tokens(
            logger,
            operation="generate_search_query",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.info(f"[AI] Generated {len(queries)} search queries: {queries}")
        return AIResponse(
            content=json.dumps(queries),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw_response=response,
        )

    async def pick_relevant_links(
        self,
        links: list[dict],
        prompt_extract: str,
        homepage_url: str,
        max_links: int = 3,
    ) -> AIResponse:
        """
        Given a list of {text, url} links from a homepage and an extraction goal,
        ask the LLM to pick the URLs most likely to contain the answer.
        Returns AIResponse with JSON {"urls": [...]} in content.
        """
        if not links:
            return AIResponse(content='{"urls": []}', input_tokens=0, output_tokens=0, model=self.model)

        # Cap input: a homepage rarely has >150 useful links; beyond that we're sending noise
        capped = links[:150]
        links_block = "\n".join(f"- {l['text']} → {l['url']}" for l in capped)

        user_message = PICK_RELEVANT_LINKS_USER_PROMPT.format(
            prompt_extract=prompt_extract,
            homepage_url=homepage_url,
            links_block=links_block,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PICK_RELEVANT_LINKS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            extra_headers={
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
            extra_body=self.extra_body,
        )

        raw = response.choices[0].message.content or '{"urls": []}'
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        log_tokens(
            logger,
            operation="pick_relevant_links",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Robust parse: accept {"urls": [...]}, bare list, or JSON-inside-markdown
        urls: list[str] = []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            import re as _re
            match = _re.search(r'\{.*\}|\[.*\]', raw, _re.DOTALL)
            parsed = json.loads(match.group(0)) if match else None

        if isinstance(parsed, dict) and isinstance(parsed.get("urls"), list):
            urls = [u for u in parsed["urls"] if isinstance(u, str)]
        elif isinstance(parsed, list):
            urls = [u for u in parsed if isinstance(u, str)]

        # Hard-filter to links that were actually in the input (no hallucinated URLs)
        allowed = {l["url"] for l in capped}
        urls = [u for u in urls if u in allowed][:max_links]

        logger.info(f"[AI] Picked {len(urls)} relevant link(s) from {len(capped)}: {urls}")
        return AIResponse(
            content=json.dumps({"urls": urls}),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw_response=response,
        )
