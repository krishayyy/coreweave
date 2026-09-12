"""One command for the demo: run the hero scenario live and export the display.

    python scripts/demo.py            # run it with the model in the loop
    python scripts/demo.py --oracle   # diagnostic: what a correct account looks like

Prints the reasoning as it happens, so the terminal is watchable while the
display renders.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop import llm, tracing                   # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402

HERO_INDEX = 9

DIM, BOLD, RED, CYAN, AMBER, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[36m", "\033[33m", "\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=HERO_INDEX)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    tracing.init()
    arm = "heuristic" if args.oracle else "llm"
    if not args.oracle and not llm.available():
        print(f"{RED}No LLM credentials. Set GROQ_API_KEY (or another provider), "
              f"or pass --oracle.{RESET}")
        return 1

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    scenario = [s for s in generate_suite(grid, 30, 24, 12, seed=7)
                if s.kind == "B"][args.index]

    if args.oracle:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ceiling_check import oracle_factory
        import searchloop.loop as loop_mod
        loop_mod.nominate_heuristic = oracle_factory(scenario, grid)

    print(f"\n{BOLD}OPERATION {scenario.id}{RESET}   "
          f"{DIM}[{scenario.subkind}]{RESET}\n")
    print(f"{scenario.case_file}\n")
    if not args.oracle:
        print(f"{DIM}model: {llm.resolve_provider().model}{RESET}")
    print(f"{DIM}{'-' * 76}{RESET}")

    def on_period(trace, belief, segment):
        for e in scenario.late_evidence:
            if e.period == trace.period:
                print(f"\n{CYAN}  NEW EVIDENCE  {e.text}{RESET}\n")
        if trace.nominations:
            print(f"\n{RED}  PREMISE FAILING{RESET}  {trace.trigger_reason}")
            for nom in trace.nominations:
                print(f"{AMBER}    -> {nom['label']}{RESET}")
                print(f"       {nom['narrative']}")
                print(f"{DIM}       start {nom['distance_km']:.1f} km on bearing "
                      f"{nom['bearing_deg']:.0f}, profile {nom['profile_key']}, "
                      f"prior {nom['prior']:.2f}{RESET}")
                for cite in nom.get("evidence_cited", [])[:2]:
                    print(f"{DIM}       cites: {cite}{RESET}")
            print()
        for rej in trace.rejected_nominations:
            print(f"{DIM}    rejected: {rej}{RESET}")
        lead = trace.leader_label[:38]
        mark = " <-- FOUND" if trace.found else ""
        print(f"  P{trace.period:02d}  {lead:<38} w={trace.leader_weight:.2f}  "
              f"{100 * trace.leader_exhaustion:3.0f}% ruled out{mark}")

    result = run_scenario(grid, pod, scenario, arm, CFG,
                          np.random.default_rng(stable_seed(scenario.id)),
                          on_period=on_period)

    print(f"{DIM}{'-' * 76}{RESET}")
    truth = f"{scenario.true_distance_km:.1f} km" if hasattr(scenario, "true_distance_km") \
        else f"{result.true_distance_km:.1f} km"
    if result.found:
        print(f"\n{BOLD}SUBJECT LOCATED{RESET} in period {result.periods_to_find}, "
              f"{truth} from the planning point.")
    else:
        print(f"\n{BOLD}NOT LOCATED{RESET} within {CFG.max_periods} periods. "
              f"Subject was {truth} from the planning point.")
    print(f"revisions {result.revisions}   "
          f"peak belief percentile at the true location {result.peak_truth_percentile:.0f}%\n")

    if not args.no_export:
        cmd = [sys.executable, str(ROOT / "scripts" / "export_frames.py"),
               "--kind", "B", "--index", str(args.index), "--arm", arm,
               "--out", str(ROOT / "web" / "public" / "run")]
        if args.oracle:
            cmd.append("--oracle")
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"{DIM}display refreshed -- "
              f"python3 -m http.server 5173 --directory web{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
