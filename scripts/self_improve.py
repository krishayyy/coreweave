"""Run the self-improvement loop: search, review your own record, rewrite your
own instructions, and keep the change only if it measurably helps.

    python scripts/self_improve.py --rounds 1

Folds are strict:

    training   scenarios the agent searches, and whose failures it reviews
    validation a disjoint set, used only to decide whether a proposed lesson
               is kept -- never the fold the lesson was derived from
    reporting  seed 7, the held-out suite the headline result comes from,
               which this script never touches at all

The acceptance gate is the point. A language model asked to improve its own
prompt will always produce something that sounds like an improvement. Whether it
is one is an empirical question, and the answer here is usually no -- which is
exactly why the gate exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import searchloop.loop as loop_mod                    # noqa: E402
from searchloop import llm, tracing                   # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.revision import _SYSTEM, _profile_menu  # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.selfimprove import (                  # noqa: E402
    LessonBook, analyse, propose_lessons, validate,
)
from searchloop.terrain import load_terrain           # noqa: E402

TRAIN_SEED = 99
DIM, BOLD, GREEN, RED, AMBER, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m")


def rule(title: str) -> None:
    print(f"\n{DIM}{'─' * 74}{RESET}")
    print(f"{BOLD}{title}{RESET}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--validate", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=3,
                    help="distinct instructions proposed per round, each validated")
    ap.add_argument("--book", default=str(ROOT / "data" / "lessons.json"))
    args = ap.parse_args()

    if not llm.available():
        print(f"{RED}No LLM credentials. Set GROQ_API_KEY (or another provider).{RESET}")
        return 1
    tracing.init()

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)

    pool = [s for s in generate_suite(grid, 0, args.train + args.validate, 0,
                                      seed=TRAIN_SEED) if s.kind == "B"]
    train, holdout = pool[:args.train], pool[args.train:]
    suite = {s.id: s for s in pool}
    book = LessonBook.load(Path(args.book))

    print(f"{BOLD}SELF-IMPROVEMENT LOOP{RESET}")
    print(f"{DIM}model {llm.resolve_provider().model}   "
          f"train {len(train)} scenarios   validate {len(holdout)}   "
          f"lessons in force {len(book.accepted)}{RESET}")
    print(f"{DIM}the reported suite (seed 7) is never touched by this script{RESET}")

    base_instructions = _SYSTEM % {
        "profiles": _profile_menu(), "extent": 20.6, "reach": 5.2,
        "near": 5.8, "far": 11.3,
    }

    for round_no in range(1, args.rounds + 1):
        rule(f"ROUND {round_no} — 1. SEARCH  (the agent works, and fails)")
        loop_mod.ACTIVE_LESSONS = book.prompt_section()
        runs = []
        for scenario in train:
            result = run_scenario(grid, pod, scenario, "llm", CFG,
                                  np.random.default_rng(stable_seed(scenario.id)))
            runs.append(result.to_dict())
            mark = f"{GREEN}found{RESET}" if result.found else f"{DIM}not found{RESET}"
            print(f"  {scenario.id} [{scenario.subkind:<21}] "
                  f"{result.revisions} revisions, {mark}", flush=True)

        rule(f"ROUND {round_no} — 2. REVIEW  (scoring its own proposals against truth)")
        report = analyse(runs, suite, grid)
        print(report.render())

        rule(f"ROUND {round_no} — 3. PROPOSE  ({args.candidates} candidate instructions)")
        candidates = propose_lessons(report, book, base_instructions, k=args.candidates)
        if not candidates:
            print(f"{RED}  no usable proposals returned{RESET}")
            continue
        for i, cand in enumerate(candidates, 1):
            print(f"{AMBER}  {i}. \"{cand.text}\"{RESET}")
            print(f"{DIM}     {cand.rationale[:180]}{RESET}")

        rule(f"ROUND {round_no} — 4. VALIDATE  (each, on {len(holdout)} scenarios "
             f"it did not learn from)")
        scored: list[tuple[float, Any]] = []
        for i, cand in enumerate(candidates, 1):
            result = validate(cand, book, holdout, grid, pod, CFG)
            cand.validation = result.to_dict()
            scored.append((result.gain_deg, cand))
            state = (f"{GREEN}+{result.gain_deg:.0f} deg{RESET}" if result.gain_deg > 0
                     else f"{RED}{result.gain_deg:+.0f} deg{RESET}")
            print(f"  {i}. {result.bearing_before:5.1f} -> {result.bearing_after:5.1f} deg"
                  f"   {state}   ({result.improved} better, {result.worsened} worse)",
                  flush=True)

        rule(f"ROUND {round_no} — 5. GATE")
        scored.sort(key=lambda x: -x[0])
        best_gain, best = scored[0]
        best.accepted = best.validation["verdict"] == "accepted"
        if best.accepted:
            print(f"{GREEN}  ACCEPTED{RESET}  \"{best.text}\"")
            print(f"  {best.validation['bearing_before']:.0f} -> "
                  f"{best.validation['bearing_after']:.0f} deg on "
                  f"{best.validation['n']} held-out scenarios; applies to every "
                  f"future search.")
        else:
            print(f"{RED}  NOTHING ACCEPTED{RESET} — best candidate: "
                  f"{best.validation['verdict']}.")
            print(f"{DIM}  All candidates recorded so they are not proposed again.{RESET}")

        book.lessons += candidates
        book.save(Path(args.book))

    rule("LESSON BOOK")
    if not book.lessons:
        print("  (empty)")
    for lesson in book.lessons:
        state = f"{GREEN}in force{RESET}" if lesson.accepted else f"{DIM}rejected{RESET}"
        print(f"  [{state}] {lesson.text}")
        if lesson.validation:
            print(f"{DIM}           {lesson.validation['bearing_before']:.0f} -> "
                  f"{lesson.validation['bearing_after']:.0f} deg on "
                  f"{lesson.validation['n']} held-out scenarios{RESET}")
    print(f"\n{DIM}written to {args.book}{RESET}\n")

    accepted = len(book.accepted)
    print(f"{BOLD}{accepted} of {len(book.lessons)} proposals survived validation.{RESET}")
    print(f"{DIM}A model asked to improve its own prompt always produces something "
          f"that sounds\nlike an improvement. The gate is what separates the ones "
          f"that are.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
