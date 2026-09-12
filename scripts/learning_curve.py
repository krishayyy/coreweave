"""Does the agent get better as it accumulates resolved cases?

Runs a sequence of searches. After each one resolves, the truth becomes known
and is written into memory. Facing the next case, the agent retrieves the most
similar resolved ones and is shown what turned out to be true.

The evidence is a difference in differences. With memory, later cases should
beat earlier ones, because later cases have more precedent to draw on. Without
memory, the same sequence should be flat -- nothing changes between case 1 and
case 24. A rising line on its own would prove nothing: the tail of any random
sequence can look easier than its head. The flat control is what makes the
rising line mean something.

Run on a fresh seed: neither the tuning fold nor the suite the headline result
is reported on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import searchloop.loop as loop_mod                    # noqa: E402
from searchloop import llm, tracing                   # noqa: E402
from searchloop.casememory import CaseMemory, record_from  # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402

CURVE_SEED = 123
DIM, BOLD, GREEN, RED, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[36m", "\033[0m")


def best_bearing_error(result, scenario, grid) -> float | None:
    """How close the best proposal came to the true starting point."""
    dr = scenario.true_anchor_rc[0] - scenario.ipp_rc[0]
    dc = scenario.true_anchor_rc[1] - scenario.ipp_rc[1]
    true_bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
    errs = [abs((n["bearing_deg"] - true_bearing + 180) % 360 - 180)
            for t in result.trace for n in t.nominations]
    return min(errs) if errs else None


def run_sequence(scenarios, grid, pod, use_memory: bool, label: str):
    memory = CaseMemory() if use_memory else None
    loop_mod.ACTIVE_MEMORY = memory
    rows = []
    print(f"\n{BOLD}{label}{RESET}")
    for i, scenario in enumerate(scenarios, 1):
        result = run_scenario(grid, pod, scenario, "llm", CFG,
                              np.random.default_rng(stable_seed(scenario.id)))
        err = best_bearing_error(result, scenario, grid)
        rows.append({"i": i, "id": scenario.id, "subkind": scenario.subkind,
                     "found": bool(result.found), "bearing_error": err,
                     "memory_size": len(memory) if memory else 0})
        if memory is not None:
            memory.add(record_from(scenario, grid, result.found))
        mark = f"{GREEN}found{RESET}" if result.found else f"{DIM}  -  {RESET}"
        errtxt = f"{err:5.0f} deg" if err is not None else "   --  "
        print(f"  {i:2d}. {scenario.id} [{scenario.subkind:<21}] "
              f"precedent={rows[-1]['memory_size']:2d}  best bearing {errtxt}  {mark}",
              flush=True)
    loop_mod.ACTIVE_MEMORY = None
    return rows


def halves(rows, key):
    mid = len(rows) // 2
    early = [r[key] for r in rows[:mid] if r[key] is not None]
    late = [r[key] for r in rows[mid:] if r[key] is not None]
    return float(np.mean(early)) if early else float("nan"), \
        float(np.mean(late)) if late else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "runs" / "learning_curve.json"))
    ap.add_argument("--control", action="store_true", default=True)
    args = ap.parse_args()

    if not llm.available():
        print(f"{RED}No LLM credentials.{RESET}")
        return 1
    tracing.init()

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    scenarios = [s for s in generate_suite(grid, 0, args.n, 0, seed=CURVE_SEED)
                 if s.kind == "B"][:args.n]

    print(f"{BOLD}LEARNING CURVE{RESET}  {DIM}{len(scenarios)} cases, seed {CURVE_SEED} "
          f"(neither the tuning fold nor the reported suite){RESET}")

    with_mem = run_sequence(scenarios, grid, pod, True,
                            "WITH MEMORY — each resolved case is written down")
    without = run_sequence(scenarios, grid, pod, False,
                           "CONTROL — identical cases, nothing remembered")

    print(f"\n{DIM}{'─' * 74}{RESET}")
    print(f"{BOLD}DID IT GET BETTER?{RESET}\n")
    print(f"{'':<16}{'first half':>14}{'second half':>14}{'change':>12}")
    for label, rows in (("with memory", with_mem), ("control", without)):
        e, l = halves(rows, "bearing_error")
        fe = np.mean([r["found"] for r in rows[:len(rows) // 2]])
        fl = np.mean([r["found"] for r in rows[len(rows) // 2:]])
        print(f"  {label:<14}{e:>10.0f} deg{l:>10.0f} deg{l - e:>+9.0f} deg")
        print(f"  {'':<14}{100 * fe:>10.0f} %{100 * fl:>10.0f} %{100 * (fl - fe):>+9.0f} pp"
              f"   {DIM}(find rate){RESET}")

    we, wl = halves(with_mem, "bearing_error")
    ce, cl = halves(without, "bearing_error")
    did = (wl - we) - (cl - ce)
    print(f"\n  difference in differences: {did:+.0f} deg")
    print(f"  {DIM}negative means memory improved bearing beyond what the "
          f"sequence alone explains{RESET}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"seed": CURVE_SEED, "with_memory": with_mem, "control": without}, indent=2))
    print(f"\n{DIM}wrote {args.out}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
