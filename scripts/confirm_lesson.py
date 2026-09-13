"""Measure an accepted lesson on a fold it was never selected on.

    python scripts/confirm_lesson.py --lesson lesson-003 --n 12

The self-improvement loop proposes k candidates, measures all of them on one
validation fold, and adopts the best. That makes the adopted candidate's gain
the MAXIMUM of k noisy estimates, which is biased upward by construction --
the same reason the n=6 run crowned a candidate that measured -1 degrees at
n=12. Re-reading the selection fold cannot detect this, however many times it
is re-read: the number that comes back is the number the candidate was chosen
for.

The only honest estimate comes from a fold that played no part in the choice.
This script builds one from a different seed, measures the adopted lesson
against no lesson on it, paired, and reports a bootstrap interval.

Folds in play:

    seed 99    training + validation for the self-improvement loop
    seed 7     the reported suite, which nothing here ever touches
    seed 4242  this confirmation fold, used for nothing else

A gain that survives here is worth quoting. One that does not means the loop
selected noise, which is a finding about the loop rather than about the lesson.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop import llm, tracing                   # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.scenario import generate_suite        # noqa: E402
from searchloop.selfimprove import (                  # noqa: E402
    LessonBook, probe_bearing_error,
)
from searchloop.terrain import load_terrain           # noqa: E402

CONFIRM_SEED = 4242
DIM, BOLD, GREEN, RED, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[0m")


def bootstrap_ci(diffs: list[float], reps: int = 20000, seed: int = 0
                 ) -> tuple[float, float, float]:
    """Paired bootstrap over per-scenario differences: (lo, hi, two-sided p)."""
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    lo, hi = means[int(0.025 * reps)], means[int(0.975 * reps)]
    n_ge = sum(m >= 0 for m in means)
    p = 2 * min(n_ge, reps - n_ge) / reps
    return lo, hi, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default=None,
                    help="lesson id to confirm; default is the accepted one")
    ap.add_argument("--n", type=int, default=12, help="confirmation fold size")
    ap.add_argument("--seed", type=int, default=CONFIRM_SEED)
    ap.add_argument("--book", default=str(ROOT / "data" / "lessons.json"))
    args = ap.parse_args()

    if not llm.available():
        print(f"{RED}No LLM credentials.{RESET}")
        return 1
    tracing.init()

    book = LessonBook.load(Path(args.book))
    if args.lesson:
        chosen = [x for x in book.lessons if x.id == args.lesson]
    else:
        chosen = book.accepted
    if not chosen:
        print(f"{RED}No such lesson in {args.book}.{RESET}")
        return 1
    lesson = chosen[0]

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    fold = [s for s in generate_suite(grid, 0, args.n, 0, seed=args.seed)
            if s.kind == "B"][:args.n]

    print(f"{BOLD}CONFIRMATION FOLD{RESET}")
    print(f"{DIM}lesson  {lesson.id}: {lesson.text[:96]}...{RESET}")
    print(f"{DIM}selected on the seed-99 fold with a measured gain of "
          f"{lesson.validation.get('gain_deg', float('nan')):+.1f} deg{RESET}")
    print(f"{DIM}measured here on seed {args.seed}, {len(fold)} scenarios it "
          f"played no part in selecting{RESET}\n")

    # The baseline is the book as it stood BEFORE this lesson was adopted, so
    # the contrast isolates this lesson rather than the whole book.
    prior = [x for x in book.accepted if x.id != lesson.id]
    base_lessons = LessonBook.render(prior)
    trial_lessons = LessonBook.render(prior + [lesson])

    before: list[float] = []
    after: list[float] = []
    kinds: list[str] = []
    for i, scenario in enumerate(fold, 1):
        b = probe_bearing_error(scenario, grid, pod, CFG, base_lessons)
        a = probe_bearing_error(scenario, grid, pod, CFG, trial_lessons)
        if not b or not a:
            print(f"  {scenario.id} [{scenario.subkind:<21}] "
                  f"{DIM}no usable proposal on one arm, skipped{RESET}", flush=True)
            continue
        before.append(min(b))
        after.append(min(a))
        kinds.append(scenario.subkind)
        delta = after[-1] - before[-1]
        mark = GREEN if delta < -1 else (RED if delta > 1 else DIM)
        print(f"  {scenario.id} [{scenario.subkind:<21}] "
              f"{before[-1]:5.1f} -> {after[-1]:5.1f} deg   "
              f"{mark}{-delta:+5.1f}{RESET}", flush=True)

    if len(before) < 3:
        print(f"\n{RED}Only {len(before)} usable scenarios; not enough to "
              f"conclude anything.{RESET}")
        return 1

    diffs = [a - b for a, b in zip(after, before)]     # negative == improvement
    gain = -st.mean(diffs)
    lo, hi, p = bootstrap_ci(diffs)
    improved = sum(d < -1 for d in diffs)
    worsened = sum(d > 1 for d in diffs)

    print(f"\n{BOLD}RESULT{RESET}")
    print(f"  before            {st.mean(before):5.1f} deg")
    print(f"  after             {st.mean(after):5.1f} deg")
    print(f"  gain              {gain:+5.1f} deg   "
          f"95% CI [{-hi:+.1f}, {-lo:+.1f}]   p = {p:.3f}")
    print(f"  scenarios         {improved} better, {worsened} worse, "
          f"{len(diffs) - improved - worsened} unchanged  (n={len(diffs)})")

    # A confirmation fold drawn from a fresh seed does not inherit the mix of
    # failure modes the lesson was selected on. If it happens to be enriched for
    # the category the lesson helps most, the overall mean flatters it. Break the
    # result out so the composition cannot hide inside the average.
    print(f"\n{BOLD}BY FAILURE MODE{RESET}")
    for kind in sorted(set(kinds)):
        idx = [j for j, k in enumerate(kinds) if k == kind]
        kb = st.mean(before[j] for j in idx)
        ka = st.mean(after[j] for j in idx)
        print(f"  {kind:<22} {kb:5.1f} -> {ka:5.1f} deg   {kb - ka:+5.1f}   (n={len(idx)})")
    mix = "  ".join(f"{k}={v}" for k, v in sorted(Counter(kinds).items()))
    print(f"{DIM}  this fold is {mix}. The selection fold had a different mix, so "
          f"the overall mean above is not comparable to the selection gain "
          f"without reading these rows.{RESET}")

    selected_gain = lesson.validation.get("gain_deg")
    if selected_gain is not None:
        print(f"\n{DIM}  selection fold said {selected_gain:+.1f} deg; "
              f"this fold says {gain:+.1f} deg. The difference between them is "
              f"the selection bias.{RESET}")

    holds = p < 0.05 and gain > 0
    print(f"\n{(GREEN + '  HOLDS UP') if holds else (RED + '  DOES NOT HOLD UP')}"
          f"{RESET} on a fold it was not selected on.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
