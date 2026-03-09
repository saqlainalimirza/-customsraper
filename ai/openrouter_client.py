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
)

logger = setup_logger(__name__)


class OpenRouterClient(AIClient):
    """OpenRouter client that supports both GPT and Claude models."""
    
    def __init__(self, model_type: str = "gpt"):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )
        
        if model_type == "claude":
            self.model = self.settings.claude_model
        else:
            self.model = self.settings.gpt_model
        
        self.model_type = model_type

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
        
        max_chars = 100000 if "gpt" in self.model.lower() else 150000
        if len(combined_content) > max_chars:
            logger.warning(f"Content too long ({len(combined_content)} chars), truncating to {max_chars}")
            combined_content = combined_content[:max_chars] + "\n... [truncated]"
        
        user_message = EXTRACT_ANSWER_USER_PROMPT.format(
            prompt=prompt,
            content=combined_content,
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
        )
        
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
        prompt_filter: str = "",
    ) -> AIResponse:
        """Generate a web search query from input data + extraction goal."""
        data_block = "\n".join(f"{k}: {v}" for k, v in data.items())
        user_message = GENERATE_SEARCH_QUERY_USER_PROMPT.format(
            data_block=data_block,
            prompt_filter=prompt_filter or "any relevant pages",
            prompt_extract=prompt_extract,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATE_SEARCH_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "https://scaletopia.com",
                "X-Title": "Scaletopia Web Scraper",
            },
        )

        content = (response.choices[0].message.content or "").strip().strip('"')
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        log_tokens(
            logger,
            operation="generate_search_query",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.info(f"[AI] Generated search query: '{content}'")
        return AIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            raw_response=response,
        )
