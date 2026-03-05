import json
from anthropic import AsyncAnthropic

from config import get_settings
from utils.logging import setup_logger, log_tokens
from .base import AIClient, AIResponse
from .prompts import (
    URL_FILTER_SYSTEM_PROMPT,
    URL_FILTER_USER_PROMPT,
    EXTRACT_ANSWER_SYSTEM_PROMPT,
    EXTRACT_ANSWER_USER_PROMPT,
)

logger = setup_logger(__name__)


class AnthropicClient(AIClient):
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        self.model = self.settings.anthropic_model

    async def filter_urls(
        self,
        urls: list[str],
        prompt: str,
        domain: str,
    ) -> AIResponse:
        """Filter URLs using Anthropic Claude."""
        logger.info(f"Filtering {len(urls)} URLs with Anthropic ({self.model})")
        
        urls_text = "\n".join(urls)
        user_message = URL_FILTER_USER_PROMPT.format(
            domain=domain,
            prompt=prompt,
            urls=urls_text,
        )
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=URL_FILTER_SYSTEM_PROMPT + "\n\nIMPORTANT: Return your response as a JSON object with a 'urls' key containing the array of URLs.",
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        
        content = response.content[0].text if response.content else "[]"
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
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
        """Extract answer from scraped content using Anthropic Claude."""
        logger.info(f"Extracting answer from {len(scraped_content)} pages with Anthropic ({self.model})")
        
        content_parts = []
        for url, text in scraped_content.items():
            content_parts.append(f"=== URL: {url} ===\n{text}\n")
        
        combined_content = "\n".join(content_parts)
        
        if len(combined_content) > 150000:
            logger.warning(f"Content too long ({len(combined_content)} chars), truncating")
            combined_content = combined_content[:150000] + "\n... [truncated]"
        
        user_message = EXTRACT_ANSWER_USER_PROMPT.format(
            prompt=prompt,
            content=combined_content,
        )
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=EXTRACT_ANSWER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
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
