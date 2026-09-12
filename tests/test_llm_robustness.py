"""The failure modes that actually occurred, pinned so they cannot recur.

Both of these silently corrupted a result before they were caught: a truncated
response lost an entire nomination, and rate limiting was filed as "the model
proposed nothing" -- which is a much more flattering claim than "the request
never arrived".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path as _P                               # noqa: E402
from searchloop import llm                                    # noqa: E402


def test_salvages_complete_objects_from_a_truncated_array():
    text = '[{"label":"a","d":1},{"label":"b","d":2},{"label":"c","d":'
    out = llm.extract_json(text)
    assert [o["label"] for o in out] == ["a", "b"]


def test_truncation_inside_a_string_does_not_corrupt_earlier_objects():
    text = '[{"label":"a"},{"label":"unterminated'
    assert llm.extract_json(text) == [{"label": "a"}]


def test_braces_inside_strings_are_not_treated_as_structure():
    text = '[{"label":"a }{ b","d":1}]'
    assert llm.extract_json(text)[0]["label"] == "a }{ b"


def test_escaped_quote_inside_a_string_is_handled():
    text = r'[{"label":"say \"hi\"","d":1},{"label":"b"'
    assert llm.extract_json(text)[0]["label"] == 'say "hi"'


def test_no_json_at_all_raises():
    with pytest.raises(ValueError):
        llm.extract_json("I am afraid I cannot help with that.")


def test_rate_limit_is_retried_rather_than_surfaced_as_no_output(monkeypatch):
    """A 429 must never reach the caller as an empty nomination."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_MIN_INTERVAL_S", 0.0)
    calls = {"n": 0}

    class Resp:
        def __init__(self, code, body=None):
            self.status_code = code
            self.headers = {"Retry-After": "0"}
            self._body = body

        def json(self):
            return self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            return Resp(429)
        return Resp(200, {"choices": [{"message": {"content": '[{"ok":true}]'}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.complete("s", "u", use_cache=False)
    assert out == '[{"ok":true}]'
    assert calls["n"] == 3, "must have retried through the 429s"


def test_persistent_rate_limiting_raises_loudly(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(llm, "_MAX_RETRIES", 2)

    class Resp:
        status_code = 429
        headers = {"Retry-After": "0"}

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match="did not answer"):
        llm.complete("s", "u", use_cache=False)


def test_no_credentials_raises_a_named_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                "WANDB_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not llm.available()
    with pytest.raises(llm.NoProviderError):
        llm.complete("s", "u", use_cache=False)


def test_tracing_never_prompts_without_credentials(monkeypatch):
    """weave.init() blocks on stdin when unauthenticated.

    During a live demo that is a frozen terminal, so init must decline before
    ever calling it rather than relying on it to fail fast.
    """
    from searchloop import tracing

    monkeypatch.setattr(tracing, "_INITIALISED", False)
    monkeypatch.setattr(tracing, "_WEAVE", None)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tracing, "_netrc_text", lambda: "")
    monkeypatch.setattr(_P, "exists", lambda self: False)

    called = {"n": 0}

    def explode(*a, **k):
        called["n"] += 1
        raise AssertionError("weave.init must not be reached without credentials")

    import sys as _sys
    import types
    fake = types.ModuleType("weave")
    fake.init = explode
    monkeypatch.setitem(_sys.modules, "weave", fake)

    assert tracing.init("x") is False
    assert called["n"] == 0
    assert tracing.enabled() is False
