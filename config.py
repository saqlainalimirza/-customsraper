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

    # Direct Google Gemini API key (AI Studio). Add GEMINI_API_KEY to .env.
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Jina Reader tuning (env-overridable: JINA_TOKEN_BUDGET, JINA_TIMEOUT)
    # Token budget is a CIRCUIT BREAKER — a page bigger than this FAILS the fetch
    # (not truncates). Keeps monster pages from blowing the Gemini TPM. Generous
    # default so normal product pages pass; only giant pages get rejected.
    jina_token_budget: int = 60000
    jina_timeout: int = 15  # seconds Jina waits for the page to load
    # Strip noise but KEEP nav (link discovery) + footer (B2B/wholesale/"powered
    # by" signals). Removes videos, cookie banners, popups, newsletter modals.
    jina_remove_selector: str = (
        "video,iframe,.cookie,#cookie-banner,[aria-label*=cookie],"
        ".modal-popup,.newsletter-popup,.popup-overlay,[class*=cookie-consent]"
    )

    # LangGraph agent tuning (env: AGENT_CONCURRENCY, AGENT_TIMEOUT_SECONDS).
    # Concurrency must be high enough to match Clay's burst size, or queued rows
    # blow their timeout while WAITING for a slot. Raise if you see mass timeouts;
    # lower only if you start hitting Gemini TPM again.
    agent_concurrency: int = 50
    agent_timeout_seconds: int = 120

    gpt_model: str = "openai/gpt-4o-mini"
    claude_model: str = "anthropic/claude-3.5-sonnet"
    gemini_model: str = "gemini-3-flash-preview"  # Gemini 3 Flash (preview ID)

    default_ai_provider: str = "gemini"
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
