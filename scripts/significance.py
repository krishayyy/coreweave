"""Paired significance testing on the experiment output.

Every arm sees the same scenarios with the same detection rolls, so the arms are
paired and an unpaired interval throws away exactly the information that makes
the comparison sharp. Repeats of a scenario are also not independent samples --
they share the scenario -- so the unit of resampling here is the SCENARIO, and
repeats are averaged within it first. Treating 72 runs as 72 independent trials
would overstate significance considerably.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    data = json.loads(path.read_text())
    by_arm: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for run in data["runs"]:
        cell = by_arm[run["arm"]][run["scenario_kind"]]
        cell[run["scenario_id"]].append(run)
    return by_arm


def per_scenario(cell: dict[str, list], field) -> dict[str, float]:
    """Average the repeats within each scenario before comparing arms."""
    return {sid: float(np.mean([field(r) for r in runs])) for sid, runs in cell.items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 20000,
                     seed: int = 0) -> tuple[float, tuple[float, float], float]:
    """Bootstrap the paired difference b - a, resampling scenarios."""
    ids = sorted(set(a) & set(b))
    diff = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    draws = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Two-sided p against no difference, by the sign of the bootstrap mass.
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return observed, (float(lo), float(hi)), float(min(p, 1.0))


def main() -> int:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/experiment_full.json")
    by_arm = load(path)

    treatment = sys.argv[2] if len(sys.argv) > 2 else (
        "llm" if "llm" in by_arm else "jev")
    metrics = {
        "find rate": lambda r: float(r["found"]),
        "localised": lambda r: float(r["periods_to_localize"] is not None),
    }
    labels = {"none": "library only", "heuristic": "blind relocation",
              "llm": "case-file nomination", "jev": "System One"}

    print(f"paired on scenarios, repeats averaged within scenario")
    print(f"resampling unit: scenario (not run)\n")

    for kind, title in (("B", "Type B -- wrong about WHERE"),
                        ("A", "Type A -- premise correct (the null)"),
                        ("C", "Type C -- wrong about WHO")):
        if treatment not in by_arm or kind not in by_arm[treatment]:
            continue
        n_scen = len(by_arm[treatment][kind])
        print(f"{title}   ({n_scen} scenarios)")
        for name, field in metrics.items():
            for base in ("none", "heuristic"):
                if base == treatment:
                    continue
                a = per_scenario(by_arm[base][kind], field)
                b = per_scenario(by_arm[treatment][kind], field)
                d, (lo, hi), p = paired_bootstrap(a, b)
                sig = "significant" if lo > 0 or hi < 0 else "not significant"
                print(f"  {name:<10} {treatment} vs {labels[base]:<21} "
                      f"{100 * d:+6.1f}pp  95% CI [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  "
                      f"p={p:.3f}  {sig}")
        print()

    print("A confidence interval that straddles zero means this suite cannot")
    print("distinguish the arms on that metric. It is reported either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
