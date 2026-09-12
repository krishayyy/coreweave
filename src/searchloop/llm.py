"""Provider-agnostic LLM adapter.

The loop must run with no key at all -- the deterministic arm is the baseline and
must never depend on a network call -- so this resolves whatever credentials
exist at import time and reports honestly when there are none.

Set exactly one of:
    ANTHROPIC_API_KEY
    GROQ_API_KEY       (+ GROQ_MODEL)
    OPENAI_API_KEY
    WANDB_API_KEY      (+ WANDB_BASE_URL, WANDB_MODEL)
    TYPESAFE_API_KEY   (+ TYPESAFE_BASE_URL, TYPESAFE_MODEL)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Responses are cached on disk by prompt digest. Two reasons: the experiment
# repeats scenarios, so identical prompts recur and re-paying for them is waste;
# and a cached arm reproduces exactly, which the deterministic arms already do.
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"

# Shared hosts rate-limit aggressively. Without a floor between calls the loop
# silently degrades into a stream of 429s that get swallowed as failed
# nominations -- which looks like "the model had no ideas" rather than "the
# request never arrived".
_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL", "1.1"))
_MAX_RETRIES = 5
_lock = threading.Lock()
_last_call = 0.0


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
    if key := os.getenv("GROQ_API_KEY"):
        return Provider("groq", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                        "https://api.groq.com/openai/v1/chat/completions", key, "openai")
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


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def complete(
    system: str,
    user: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> str:
    """One completion from whichever provider is configured.

    Retries on rate limiting and transient server errors. A 429 that is allowed
    to propagate reads downstream as a model that produced nothing, which is a
    materially different and much more flattering claim than the truth.
    """
    provider = resolve_provider()
    if provider is None:
        raise NoProviderError(
            "No LLM credentials found. Set ANTHROPIC_API_KEY, GROQ_API_KEY, "
            "OPENAI_API_KEY, WANDB_API_KEY or TYPESAFE_API_KEY."
        )

    key = hashlib.sha256(
        "\x00".join([provider.name, provider.model, system, user,
                     str(max_tokens), str(temperature)]).encode()
    ).hexdigest()
    cached = CACHE_DIR / f"{key}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())["text"]

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

    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            resp = requests.post(provider.base_url, headers=headers,
                                 json=payload, timeout=120)
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(2**attempt, 20))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            # Honour Retry-After when the server sends one; it knows better
            # than any backoff schedule guessed here.
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() \
                else min(2**attempt + 1, 30)
            last = requests.HTTPError(f"{resp.status_code} from {provider.name}")
            time.sleep(delay)
            continue

        resp.raise_for_status()
        body = resp.json()
        text = ("".join(b.get("text", "") for b in body["content"])
                if provider.style == "anthropic"
                else body["choices"][0]["message"]["content"])

        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"provider": provider.name,
                                          "model": provider.model, "text": text}))
        return text

    raise RuntimeError(
        f"{provider.name} did not answer after {_MAX_RETRIES} attempts: {last}"
    )


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a completion, fenced or bare.

    Tolerates truncation. A response cut off by the token budget still usually
    contains several complete objects, and salvaging those is much better than
    discarding the whole nomination -- during a live search a dropped proposal
    means the search does not get revised at all.
    """
    if fence := re.search(r"```(?:json)?\s*(.+?)```", text, re.S):
        text = fence.group(1)
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in completion: {text[:200]!r}")

    body = text[start:]
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(body)
        return value
    except json.JSONDecodeError:
        pass

    # Truncated. Walk the string and keep every top-level object that closed.
    salvaged: list[Any] = []
    depth = 0
    obj_start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                try:
                    salvaged.append(json.loads(body[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = -1

    if not salvaged:
        raise ValueError(f"no complete JSON object in completion: {text[:200]!r}")
    return salvaged
