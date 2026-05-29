"""
AI Debate Board — Venice API Client

Handles authentication, rate limiting, and API calls to Venice.ai
for multi-agent debate orchestration.
"""

import os
import json
import time
import yaml
import requests
from typing import Optional
from dataclasses import dataclass, field


VENICE_BASE = "https://api.venice.ai/api/v1"
CHAT_ENDPOINT = f"{VENICE_BASE}/chat/completions"


@dataclass
class VeniceConfig:
    api_key: str = ""
    primary_model: str = "zai-org-glm-5-1"
    secondary_model: str = "grok-4-20"
    tertiary_model: str = "deepseek-v4-pro"
    max_retries: int = 3
    base_delay: float = 2.0
    timeout: int = 120

    @classmethod
    def from_env_and_hermes(cls) -> "VeniceConfig":
        """Load config from env var or Hermes config.yaml."""
        api_key = os.environ.get("VENICE_API_KEY", "")
        if not api_key:
            # Try Hermes config
            hermes_cfg = os.path.expanduser("~/.hermes/config.yaml")
            if os.path.exists(hermes_cfg):
                with open(hermes_cfg) as f:
                    cfg = yaml.safe_load(f)
                # Primary: model.api_key
                api_key = cfg.get("model", {}).get("api_key", "")
                # Fallback: custom_providers[0].api_key
                if not api_key and cfg.get("custom_providers"):
                    api_key = cfg["custom_providers"][0].get("api_key", "")
                # Last resort: top-level
                if not api_key:
                    api_key = cfg.get("api_key", "")

        if not api_key:
            raise ValueError(
                "No Venice API key found. Set VENICE_API_KEY env var "
                "or ensure ~/.hermes/config.yaml has model.api_key"
            )

        return cls(api_key=api_key)


def venice_chat(
    config: VeniceConfig,
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_completion_tokens: int = 2048,
    response_format: Optional[dict] = None,
    venice_params: Optional[dict] = None,
) -> dict:
    """
    Call Venice chat/completions with retry and fallback.

    Returns the full API response dict.
    """
    model = model or config.primary_model

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }

    if response_format:
        payload["response_format"] = response_format

    if venice_params:
        payload["venice_parameters"] = venice_params
    else:
        # Disable Venice system prompt for full control of agent prompts
        payload["venice_parameters"] = {
            "include_venice_system_prompt": False,
            "disable_thinking": True,
            "strip_thinking_response": True,
        }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    # Build model fallback chain
    model_chain = [model]
    if model != config.secondary_model:
        model_chain.append(config.secondary_model)
    if model != config.tertiary_model:
        model_chain.append(config.tertiary_model)

    last_error = None
    for attempt_model in model_chain:
        payload["model"] = attempt_model
        for attempt in range(config.max_retries):
            try:
                resp = requests.post(
                    CHAT_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=config.timeout,
                )

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    # Rate limit — back off and retry
                    delay = config.base_delay * (2 ** attempt)
                    print(f"  ⏳ Rate limited on {attempt_model}, backing off {delay}s...")
                    time.sleep(delay)
                    continue

                if resp.status_code in (500, 503, 504):
                    # Transient error — retry
                    delay = config.base_delay * (2 ** attempt)
                    print(f"  ⚠️ Server error {resp.status_code} on {attempt_model}, retry {attempt+1}...")
                    time.sleep(delay)
                    continue

                if resp.status_code == 404:
                    # Model not found — try next in chain
                    error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    suggested = error_data.get("error", {}).get("message", "")
                    print(f"  ❌ Model {attempt_model} not found. {suggested}")
                    last_error = f"Model {attempt_model} not found: {suggested}"
                    break  # Break retry loop, try next model

                # Non-retryable error
                print(f"  ❌ API error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()

            except requests.exceptions.Timeout:
                delay = config.base_delay * (2 ** attempt)
                print(f"  ⏱️ Timeout on {attempt_model}, retry {attempt+1}...")
                time.sleep(delay)
                continue

        last_error = f"Exhausted retries for {attempt_model}"

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def extract_content(response: dict) -> str:
    """Extract the text content from a Venice chat response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def extract_usage(response: dict) -> dict:
    """Extract token usage from a Venice chat response."""
    try:
        return response.get("usage", {})
    except (KeyError, IndexError):
        return {}
