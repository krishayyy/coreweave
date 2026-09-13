"""The gate, and the three ways it has silently stopped being a gate.

Each test here corresponds to a defect that shipped and was only caught by
reading output that looked plausible: a report that quietly contained the
answer, a gate comparing two point estimates to a fixed bar, and a duplicate
guard that any nearby value walked straight past.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop import tunable                                  # noqa: E402
from searchloop.selftune import (                               # noqa: E402
    Score, Trial, TuningLog, judge, paired_stderr,
)


def score(runs, find_rate=0.6, n=12):
    runs = list(runs)
    return Score(find_rate, 0.5, 50.0, float(np.mean(runs)), n,
                 max(1, len(runs) // n), runs)


def trial():
    return Trial("t", "exhaustion_trigger", 0.7, 0.4, "", "", "")


# --- the report must not contain the answer ------------------------------

def test_describe_reports_the_live_value_not_the_tuned_default():
    """Under --from-naive this printed the eight hand-tuned values as
    `currently ...`, which is the result the run exists to rediscover."""
    text = tunable.describe(tunable.NAIVE)
    for name, value in tunable.NAIVE.items():
        assert f"{name}: currently {value}," in text
    for name, value in tunable.DEFAULTS.items():
        if value != tunable.NAIVE[name]:
            assert f"currently {value}," not in text


def test_describe_omits_values_when_given_none():
    assert "currently" not in tunable.describe()


# --- the gate must be able to see its own noise --------------------------

def test_paired_stderr_is_zero_without_usable_vectors():
    """No confidence is better than invented confidence."""
    assert paired_stderr(Score(0, 0, 0, 0, 12), Score(0, 0, 0, 0, 12)) == 0.0
    assert paired_stderr(score([0.1] * 12), score([0.1] * 6)) == 0.0


def test_a_change_that_does_nothing_is_usually_refused():
    """The old gate kept a third of pure-noise changes; this one keeps far
    fewer. Not zero -- the bar is one standard error, deliberately."""
    rng = np.random.default_rng(7)
    kept = 0
    for _ in range(200):
        base = rng.random(72) * 0.6
        after = base + rng.normal(0.0, 0.30, 72)      # truly no effect
        t = trial()
        judge(t, score(base), score(after))
        kept += t.accepted
    assert kept < 60, f"kept {kept}/200 changes that do nothing"


def test_a_gain_inside_one_standard_error_is_named_as_such():
    rng = np.random.default_rng(1)
    base = rng.random(72) * 0.6
    t = trial()
    judge(t, score(base), score(base + rng.normal(0.02, 0.40, 72)))
    if not t.accepted:
        assert "bar" in t.verdict or "standard error" in t.verdict


def test_a_real_gain_survives():
    base = np.random.default_rng(2).random(72) * 0.6
    t = trial()
    judge(t, score(base, 0.60), score(base + 0.05, 0.62))
    assert t.accepted and "+0.050" in t.verdict


def test_finding_fewer_people_is_never_traded_for_the_objective():
    base = np.random.default_rng(3).random(72) * 0.6
    t = trial()
    judge(t, score(base, 0.70), score(base + 0.20, 0.55))
    assert not t.accepted and "find rate" in t.verdict


def test_the_score_knows_how_many_runs_it_averaged():
    s = score(np.zeros(72))
    assert s.n == 12 and s.repeats == 6 and s.n_runs == 72


def test_the_log_does_not_carry_the_per_run_vector():
    import json
    d = score(np.zeros(72)).to_dict()
    assert "runs" not in d
    json.dumps(d)


# --- the starting point must be recorded, not guessed --------------------

def test_baseline_round_trips(tmp_path):
    log = TuningLog(settings=dict(tunable.NAIVE), baseline=dict(tunable.NAIVE))
    log.settings["max_revisions"] = 3
    log.save(tmp_path / "t.json")
    assert TuningLog.load(tmp_path / "t.json").baseline["max_revisions"] == 1


def test_a_log_without_a_baseline_reports_no_phantom_progress(tmp_path):
    """Older logs predate the field. Assume they started where they sit,
    rather than diffing naive settings against DEFAULTS and calling the eight
    differences things the loop achieved."""
    import json
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"settings": tunable.NAIVE, "trials": []}))
    log = TuningLog.load(p)
    assert log.baseline == log.settings


@pytest.mark.parametrize("name,knob", tunable.KNOBS.items())
def test_every_knob_has_a_naive_value_inside_its_bounds(name, knob):
    assert name in tunable.NAIVE
    assert knob.lo <= tunable.NAIVE[name] <= knob.hi
    assert knob.lo <= knob.default <= knob.hi


# --- the duplicate guard must not be walked past -------------------------

def _propose_value(monkeypatch, parameter, value, prior=None):
    """Drive the real propose() with a canned model response."""
    import json as _json
    from searchloop import llm, selftune
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _json.dumps(
        {"parameter": parameter, "value": value,
         "rationale": "r", "predicted_effect": "p"}))
    log = TuningLog(settings=dict(tunable.NAIVE), baseline=dict(tunable.NAIVE))
    if prior is not None:
        t = trial()
        t.parameter, t.was, t.now = parameter, tunable.NAIVE[parameter], prior
        log.trials.append(t)
    return selftune.propose(score(np.zeros(72)), log, "BY FAILURE MODE")


def test_a_value_already_measured_is_refused(monkeypatch):
    assert _propose_value(monkeypatch, "exhaustion_trigger", 0.4, prior=0.4) is None


def test_a_value_too_close_to_one_already_measured_is_refused(monkeypatch):
    """0.499 where 0.5 was refused is not a new experiment: the difference is
    an order of magnitude inside the noise it would be measured through."""
    assert _propose_value(monkeypatch, "exhaustion_trigger", 0.41, prior=0.4) is None


def test_a_genuinely_different_value_is_allowed(monkeypatch):
    t = _propose_value(monkeypatch, "exhaustion_trigger", 0.25, prior=0.4)
    assert t is not None and t.now == 0.25


def test_proposing_the_current_value_is_not_a_proposal(monkeypatch):
    value = tunable.NAIVE["exhaustion_trigger"]
    assert _propose_value(monkeypatch, "exhaustion_trigger", value) is None


def test_an_unknown_parameter_is_refused(monkeypatch):
    assert _propose_value(monkeypatch, "search_harder", 1.0) is None
