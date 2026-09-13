"""Run the self-tuning loop: search, review, propose a change, measure, keep or revert.

    python scripts/self_tune.py --rounds 4

Folds are strict. The agent reviews its performance on a training fold, and
every proposal is measured on a disjoint validation fold. Neither is the suite
the headline result is reported on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop import llm, tracing, tunable                # noqa: E402
from searchloop.config import DEFAULT as CFG                # noqa: E402
from searchloop.grid import build_grid                      # noqa: E402
from searchloop.pod import pod_field                        # noqa: E402
from searchloop.scenario import generate_suite              # noqa: E402
from searchloop.selftune import (                           # noqa: E402
    TuningLog, judge, propose, score_fold,
)
from searchloop.terrain import load_terrain                 # noqa: E402

TUNE_SEED = 99
DIM, BOLD, GREEN, RED, AMBER, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")


def rule(t): print(f"\n{DIM}{'─' * 74}{RESET}\n{BOLD}{t}{RESET}\n")


def breakdown(grid, pod, scenarios, settings, arm, repeats):
    """Per-failure-mode performance, so the agent has something to reason from.

    Repeated exactly as many times as the headline numbers above it. This ran a
    single pass while the summary ran six, which put the noisiest table in the
    report directly under the sentence telling the model to reason from it: at
    three cases per failure mode, one pass moves a mode's located rate by 33
    points on one unlucky sweep, and the model would duly propose a change to
    explain it.
    """
    import collections
    import numpy as np
    from searchloop import tunable
    from searchloop.loop import run_scenario
    from searchloop.scenario import stable_seed

    tunable.apply(settings)
    cfg = tunable.as_config(CFG, settings)
    by = collections.defaultdict(list)
    revs = []
    for repeat in range(repeats):
        for s in scenarios:
            r = run_scenario(grid, pod, s, arm, cfg,
                             np.random.default_rng(stable_seed(s.id, repeat)))
            by[s.subkind].append((r.found, r.peak_truth_percentile,
                                  r.periods_to_localize))
            revs.append(r.revisions)
    lines = ["BY FAILURE MODE"]
    for k, v in sorted(by.items()):
        found = sum(x[0] for x in v) / len(v)
        peak = float(np.mean([x[1] for x in v]))
        lines.append(f"  {k:<22} located {100 * found:3.0f}%   "
                     f"mean best ranking {peak:3.0f}%   "
                     f"(n={len(v) // repeats} cases x {repeats})")
    lines.append(f"  mean revisions per search: {float(np.mean(revs)):.1f} "
                 f"(cap is {cfg.max_revisions})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--validate", type=int, default=12)
    ap.add_argument("--arm", default="jev")
    ap.add_argument("--repeats", type=int, default=6,
                    help="passes over each scenario, re-rolling detection only. "
                         "The gate cannot see a change smaller than the noise, "
                         "and the noise falls as 1/sqrt(repeats)")
    ap.add_argument("--from-naive", action="store_true",
                    help="start from an untuned configuration, so the loop has "
                         "somewhere to climb from")
    ap.add_argument("--reset", action="store_true", help="discard prior trials")
    args = ap.parse_args()

    if not llm.available():
        print(f"{RED}No LLM credentials for the proposing step.{RESET}")
        return 1
    tracing.init()

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    wanted = args.train + args.validate
    pool = [s for s in generate_suite(grid, 0, wanted, 0, seed=TUNE_SEED)
            if s.kind == "B"]
    if len(pool) < wanted:
        print(f"{RED}Asked for {wanted} scenarios, the generator produced "
              f"{len(pool)}. The validation fold would be silently short and "
              f"every measurement on it noisier than reported.{RESET}")
        return 1
    train, holdout = pool[:args.train], pool[args.train:]

    log = TuningLog() if args.reset else TuningLog.load()
    if args.from_naive:
        resuming = log.trials and log.baseline == tunable.NAIVE
        if log.trials and not resuming:
            # Starting somewhere new invalidates the history: `was` values, the
            # already-tried list and the duplicate guard all describe a
            # different baseline. Continuing from them mixes two experiments in
            # one log and reports the difference between them as progress.
            print(f"{RED}--from-naive, but the {len(log.trials)} recorded "
                  f"trial(s) started from a different configuration. Pass "
                  f"--reset to begin a clean experiment.{RESET}")
            return 1
        log.baseline = dict(tunable.NAIVE)
        if not resuming:
            log.settings = dict(tunable.NAIVE)
        else:
            print(f"{DIM}resuming a naive-start experiment: {len(log.trials)} "
                  f"trial(s), {len(log.accepted)} kept{RESET}")

    print(f"{BOLD}SELF-TUNING LOOP{RESET}")
    print(f"{DIM}train {len(train)} · validate {len(holdout)} · seed {TUNE_SEED} · "
          f"arm {args.arm}{RESET}")
    print(f"{DIM}the reported suite (seed 7) is never touched here{RESET}")

    for round_no in range(1, args.rounds + 1):
        rule(f"ROUND {round_no} — 1. MEASURE ITSELF")
        before_train = score_fold(grid, pod, train, log.settings, CFG, args.arm,
                                  args.repeats)
        detail = breakdown(grid, pod, train, log.settings, args.arm, args.repeats)
        print(f"  located {100 * before_train.find_rate:.0f}%   "
              f"top-decile {100 * before_train.localise_rate:.0f}%   "
              f"speed score {before_train.objective:.3f}")
        print(f"{DIM}{detail}{RESET}")

        rule(f"ROUND {round_no} — 2. PROPOSE A CHANGE TO ITSELF")
        # A repeated or malformed proposal costs a round; give it a few tries.
        attempts = 5
        trial = None
        for attempt in range(attempts):
            trial = propose(before_train, log, detail, attempt=attempt, debug=True)
            if trial is not None:
                break
        if trial is None:
            print(f"{RED}  no new proposal after {attempts} attempts -- the model "
                  f"is re-suggesting changes already measured{RESET}")
            continue
        print(f"{AMBER}  {trial.parameter}: {trial.was} -> {trial.now}{RESET}")
        print(f"{DIM}  because: {trial.rationale}{RESET}")
        print(f"{DIM}  expects: {trial.predicted}{RESET}")

        rule(f"ROUND {round_no} — 3. MEASURE IT ON {len(holdout)} UNSEEN CASES")
        before = score_fold(grid, pod, holdout, log.settings, CFG, args.arm,
                            args.repeats)
        trial_settings = {**log.settings, trial.parameter: trial.now}
        after = score_fold(grid, pod, holdout, trial_settings, CFG, args.arm,
                           args.repeats)

        print(f"  {'':<10}{'located':>10}{'top decile':>13}{'speed score':>14}")
        print(f"  {'before':<10}{100 * before.find_rate:>9.0f}%"
              f"{100 * before.localise_rate:>12.0f}%{before.objective:>14.3f}")
        print(f"  {'after':<10}{100 * after.find_rate:>9.0f}%"
              f"{100 * after.localise_rate:>12.0f}%{after.objective:>14.3f}")

        judge(trial, before, after)
        print()
        if trial.accepted:
            log.settings[trial.parameter] = trial.now
            print(f"{GREEN}  ACCEPTED{RESET} — {trial.verdict}. "
                  f"{trial.parameter} is now {trial.now} for every future search.")
        else:
            # score_fold installs whatever it measures, so the trial settings are
            # still in force. Put the accepted configuration back before the next
            # round reads it.
            tunable.apply(log.settings)
            print(f"{RED}  REVERTED{RESET} — {trial.verdict}.")
        log.trials.append(trial)
        log.save()

    rule("WHAT IT LEARNED ABOUT ITSELF")
    for t in log.trials:
        mark = f"{GREEN}kept{RESET}" if t.accepted else f"{DIM}reverted{RESET}"
        print(f"  [{mark}] {t.parameter}: {t.was} -> {t.now}   {DIM}{t.verdict}{RESET}")
    start = log.baseline
    changed = {k: v for k, v in log.settings.items() if v != start.get(k)}
    print(f"\n{BOLD}configuration moved in {len(changed)} place(s) "
          f"from where it started{RESET}")
    for k, v in changed.items():
        print(f"  {k}: {start.get(k)} -> {v}")
    print(f"\n{len(log.accepted)} of {len(log.trials)} proposals survived measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
