"""Weave instrumentation.

Optional by construction: if weave is not installed or not logged in, every
decorator here becomes a no-op and the loop runs exactly as before. The
deterministic baseline must never depend on an observability backend being
reachable.

What is worth tracing is not the search -- that is arithmetic -- but the
revision: what the evidence had ruled out, what was proposed in response, what
was rejected and why. The rejected proposals are the most informative rows,
because they show the mixture refusing a bad account rather than absorbing it.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable

_WEAVE: Any = None
_INITIALISED = False


def init(project: str | None = None) -> bool:
    """Start Weave if it is available and configured. Never raises."""
    global _WEAVE, _INITIALISED
    if _INITIALISED:
        return _WEAVE is not None
    _INITIALISED = True

    if os.getenv("SEARCHLOOP_DISABLE_WEAVE"):
        return False
    try:
        import weave                                    # type: ignore
        weave.init(project or os.getenv("WEAVE_PROJECT", "searchloop"))
        _WEAVE = weave
        return True
    except Exception:
        _WEAVE = None
        return False


def enabled() -> bool:
    return _WEAVE is not None


def op(fn: Callable) -> Callable:
    """Trace a call when Weave is live, otherwise pass it straight through.

    Resolved at call time rather than import time so `init` can be called after
    modules are imported.
    """
    wrapped: dict[str, Callable] = {}

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if _WEAVE is None:
            return fn(*args, **kwargs)
        if "f" not in wrapped:
            try:
                wrapped["f"] = _WEAVE.op()(fn)
            except Exception:
                wrapped["f"] = fn
        return wrapped["f"](*args, **kwargs)

    return inner


def log(name: str, payload: dict[str, Any]) -> None:
    """Record a standalone event. Silent when Weave is unavailable."""
    if _WEAVE is None:
        return
    try:
        _WEAVE.publish(payload, name=name)
    except Exception:
        pass
