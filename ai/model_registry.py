"""
Single source of truth for model selection — shared by the pipeline
(OpenRouterClient) AND the LangGraph agent. Maps user-typed provider aliases
to a concrete (route, model, key, base_url, extra_body) so every endpoint
accepts the same short-forms and the "mix" load-balancer.
"""
import itertools

from config import get_settings

_OR, _NATIVE = "openrouter", "native"

# canonical key → (route, openrouter_slug | None). None slug = resolved from settings.
_MODELS = {
    "gemini":           (_NATIVE, None),                          # Gemini 3 Flash, native Google
    "gemini-flash":     (_OR, "google/gemini-3-flash-preview"),
    "gemini-flashlite": (_OR, "google/gemini-3.1-flash-lite"),
    "grok":             (_OR, "x-ai/grok-4.3"),
    "gpt-4.1":          (_OR, "openai/gpt-4.1"),
    "gpt-4.1-mini":     (_OR, "openai/gpt-4.1-mini"),
    "gpt-4.1-nano":     (_OR, "openai/gpt-4.1-nano"),
    "claude-haiku":     (_OR, "anthropic/claude-3.5-haiku"),
    "gpt":              (_OR, None),                              # settings.gpt_model
    "claude":           (_OR, None),                              # settings.claude_model
}

_ALIASES = {
    "gemini": "gemini", "gemini3": "gemini", "gemini-3": "gemini", "default": "gemini", "native": "gemini", "": "gemini",
    "flash": "gemini-flash", "geminiflash": "gemini-flash", "gemini-flash": "gemini-flash", "gemini-3-flash": "gemini-flash",
    "flashlite": "gemini-flashlite", "flash-lite": "gemini-flashlite", "lite": "gemini-flashlite",
    "geminiflashlite": "gemini-flashlite", "gemini-flash-lite": "gemini-flashlite", "gemini-3.1-flash-lite": "gemini-flashlite",
    "grok": "grok", "grok4": "grok", "grok-4": "grok", "grok-4.3": "grok", "grok43": "grok",
    "gpt-4.1": "gpt-4.1", "gpt4.1": "gpt-4.1", "gpt41": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini", "gpt41mini": "gpt-4.1-mini", "gpt-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano", "gpt41nano": "gpt-4.1-nano", "gpt-nano": "gpt-4.1-nano",
    "claude-haiku": "claude-haiku", "haiku": "claude-haiku",
    "gpt": "gpt", "claude": "claude",
}

# "mix"/"openrouter"/"auto" → round-robin across these 5 (3 providers: Google,
# xAI, OpenAI). Cheap models only — dropped full GPT-4.1 ($2/$8 per 1M) for the
# nano/mini variants. Spreads load so no single rate limit bottlenecks the batch.
_MIX_POOL = [
    "gemini-flash", "gemini-flashlite", "grok",
    "gpt-4.1-mini", "gpt-4.1-nano",
]
_mix_cycle = itertools.cycle(_MIX_POOL)


def resolve(provider: str) -> dict:
    """Resolve a user-typed provider/alias into a full model config dict:
        {key, route, model, api_key, base_url, extra_body}
    'mix'/'openrouter'/'auto' round-robins the mix pool."""
    s = get_settings()
    p = (provider or "").strip().lower().replace("_", "-").replace(" ", "-")

    if p in ("mix", "openrouter", "or", "auto", "balance"):
        key = next(_mix_cycle)
    else:
        key = _ALIASES.get(p, "gemini")

    route, slug = _MODELS[key]

    if route == _NATIVE:  # native Gemini
        return {
            "key": key, "route": route, "model": s.gemini_model,
            "api_key": s.gemini_api_key, "base_url": s.gemini_base_url,
            # native Gemini: low thinking for speed
            "extra_body": {"reasoning_effort": "low"},
        }

    # OpenRouter-routed
    model = slug
    if key == "gpt":
        model = s.gpt_model
    elif key == "claude":
        model = s.claude_model
    # Disable reasoning on OpenRouter models for speed/cost (unified param)
    extra_body = None
    if key in ("grok", "gemini-flash", "gemini-flashlite"):
        extra_body = {"reasoning": {"enabled": False}}
    return {
        "key": key, "route": route, "model": model,
        "api_key": s.openrouter_api_key, "base_url": s.openrouter_base_url,
        "extra_body": extra_body,
    }
