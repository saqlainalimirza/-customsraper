from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str = Field(alias="NEXT_PUBLIC_SUPABASE_URL")
    supabase_key: str = Field(alias="NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    scrapingbee_api_key: str = ""
    jina_api_key: str = ""
    
    gpt_model: str = "openai/gpt-4o-mini"
    claude_model: str = "anthropic/claude-3.5-sonnet"
    gemini_model: str = "google/gemini-2.5-flash-preview"
    
    default_ai_provider: str = "gpt"
    max_urls_per_domain: int = 500
    request_delay_min: float = 1.0
    request_delay_max: float = 3.0
    scrapingbee_timeout_seconds: int = 45

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
