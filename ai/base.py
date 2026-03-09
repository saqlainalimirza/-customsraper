from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AIResponse:
    """Response from an AI model with token usage."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    async def filter_urls(
        self,
        urls: list[str],
        prompt: str,
        domain: str,
    ) -> AIResponse:
        """
        Filter URLs based on a prompt.
        
        Args:
            urls: List of URLs to filter
            prompt: User prompt describing what info they're looking for
            domain: The domain being scraped
            
        Returns:
            AIResponse with filtered URLs as JSON list in content
        """
        pass

    @abstractmethod
    async def extract_answer(
        self,
        scraped_content: dict[str, str],
        prompt: str,
    ) -> AIResponse:
        """
        Extract answer from scraped content based on prompt.
        
        Args:
            scraped_content: Dict mapping URL to scraped text content
            prompt: User prompt asking for specific information
            
        Returns:
            AIResponse with the extracted answer
        """
        pass

    @abstractmethod
    async def generate_search_query(
        self,
        data: dict[str, str],
        prompt_extract: str,
    ) -> AIResponse:
        """
        Generate a Jina search query from arbitrary input data.

        Args:
            data: Key-value pairs describing the company/person (name, website, linkedin, etc.)
            prompt_extract: What information we want to find

        Returns:
            AIResponse with the search query string in content
        """
        pass
