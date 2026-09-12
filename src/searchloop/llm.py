"""Provider-agnostic LLM adapter.

The loop must run with no key at all -- the deterministic arm is the baseline and
must never depend on a network call -- so this resolves whatever credentials
exist at import time and reports honestly when there are none.

Set exactly one of:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    WANDB_API_KEY      (+ WANDB_BASE_URL, WANDB_MODEL)
    TYPESAFE_API_KEY   (+ TYPESAFE_BASE_URL, TYPESAFE_MODEL)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


class NoProviderError(RuntimeError):
    """Raised when a nomination is attempted with no credentials configured."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    base_url: str
    api_key: str
    style: str          # "anthropic" | "openai"


def resolve_provider() -> Provider | None:
    """First configured provider, or None. Never raises."""
    if key := os.getenv("ANTHROPIC_API_KEY"):
        return Provider("anthropic", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
                        "https://api.anthropic.com/v1/messages", key, "anthropic")
    if key := os.getenv("OPENAI_API_KEY"):
        return Provider("openai", os.getenv("OPENAI_MODEL", "gpt-4o"),
                        "https://api.openai.com/v1/chat/completions", key, "openai")
    if key := os.getenv("WANDB_API_KEY"):
        base = os.getenv("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")
        return Provider("wandb", os.getenv("WANDB_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                        f"{base.rstrip('/')}/chat/completions", key, "openai")
    if key := os.getenv("TYPESAFE_API_KEY"):
        base = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
        return Provider("typesafe", os.getenv("TYPESAFE_MODEL", "typesafe-default"),
                        f"{base.rstrip('/')}/chat/completions", key, "openai")
    return None


def available() -> bool:
    return resolve_provider() is not None


def complete(system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """One completion from whichever provider is configured."""
    provider = resolve_provider()
    if provider is None:
        raise NoProviderError(
            "No LLM credentials found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "WANDB_API_KEY or TYPESAFE_API_KEY."
        )

    if provider.style == "anthropic":
        headers = {"x-api-key": provider.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        payload: dict[str, Any] = {
            "model": provider.model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        }
    else:
        headers = {"Authorization": f"Bearer {provider.api_key}",
                   "content-type": "application/json"}
        payload = {
            "model": provider.model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }

    resp = requests.post(provider.base_url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    body = resp.json()

    if provider.style == "anthropic":
        return "".join(b.get("text", "") for b in body["content"])
    return body["choices"][0]["message"]["content"]


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a completion, fenced or bare."""
    if fence := re.search(r"```(?:json)?\s*(.+?)```", text, re.S):
        text = fence.group(1)
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in completion: {text[:200]!r}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[start:])
    return value
