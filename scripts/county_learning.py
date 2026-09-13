"""Does a county get better at its own ground, and is what it learns local?

Three real jurisdictions with genuinely different terrain. Each has a hidden
local deviation from the published behaviour model, which the searcher never
sees -- it only ever sees the national literature. The county learns its
deviation from its own resolved cases and shares it with every drone flying
there.

The deviations below were chosen by me, and that choice is the load-bearing
assumption of this experiment, so it is stated rather than buried: two counties
depart from the literature in opposite directions, and the third does not
depart at all. The third is the important one. If the method "learns" a
correction for a county that is genuinely average, it is fitting noise, and no
amount of improvement in the other two would mean anything.

Three arms on held-out cases:

    published   the national model, what every county starts with
    home        the county's own model, fitted on its own resolved cases
    transplant  a neighbouring county's model, applied here

`home` beating `published` alone proves very little -- four free parameters can
improve almost any fit. `home` beating `transplant` is the claim: that what was
learned is true of this ground and not of ground in general.

Metric is operational rather than statistical: search cells in descending
probability order and report what fraction of the county you cover before
reaching the subject. Lower is better. It is the number an incident commander
actually cares about.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop.config import DEFAULT as CFG                          # noqa: E402
from searchloop.county import CountyModel, ResolvedCase, fit_county   # noqa: E402
from searchloop.grid import build_grid                                # noqa: E402
from searchloop.hypotheses import PROFILES, build_prior_field         # noqa: E402
from searchloop.loop import run_scenario                              # noqa: E402
from searchloop.pod import pod_field                                  # noqa: E402
from searchloop.scenario import generate_suite, stable_seed           # noqa: E402
from searchloop.terrain import load_terrain                           # noqa: E402

# Real search-and-rescue jurisdictions, chosen for genuinely different terrain:
# glaciated volcano, barren high granite, forested Appalachian ridge.
COUNTIES = {
    # Weights are in standard deviations of that county's own terrain.
    # (drainage, downhill, canopy, slope_excess)
    "Clackamas OR": dict(lat=45.3736, lon=-121.6960,
                         truth=[0.7, 0.3, 0.0, 0.0],
                         note="wet drainages are the walkable way out"),
    "Inyo CA":      dict(lat=36.5785, lon=-118.2923,
                         truth=[-0.7, -0.2, 0.0, 0.0],
                         note="dry creeks go nowhere; movement stays off them"),
    "Buncombe NC":  dict(lat=35.7649, lon=-82.2651,
                         truth=[0.0, 0.0, 0.0, 0.0],
                         note="the null: genuinely the national average"),
}

TRAIN_SIZES = [0, 2, 4, 8, 16, 32]


def world(spec):
    return build_grid(load_terrain(spec["lat"], spec["lon"], 12, 1), 4)


def truth_model(name, spec) -> CountyModel | None:
    w = np.asarray(spec["truth"], float)
    if not np.any(w):
        return None
    return CountyModel(county=name, weights=w, n_cases=1)


def fraction_swept(field_: np.ndarray, truth_rc) -> float:
    """Fraction of the county covered before reaching the subject, searching
    the highest-probability cell first."""
    flat = field_.ravel()
    target = truth_rc[0] * field_.shape[1] + truth_rc[1]
    return float((flat > flat[target]).sum() + 1) / flat.size


def cases_and_tests(grid, scenarios, n_train):
    train, tests = [], []
    for i, s in enumerate(scenarios):
        prior = build_prior_field(grid, PROFILES[s.true_profile_key], s.true_anchor_rc)
        if i < n_train:
            train.append(ResolvedCase(s.true_anchor_rc, s.true_rc, prior))
        else:
            tests.append((s, prior))
    return train, tests


def main() -> int:
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 72
    grids, suites = {}, {}
    for name, spec in COUNTIES.items():
        grids[name] = world(spec)
        # Type A only: this loop is about terrain preference, not about
        # revising a wrong premise. Mixing them would confound the two.
        suites[name] = generate_suite(grids[name], n_a=n_total, n_b=0, n_c=0,
                                      seed=11, local=truth_model(name, spec))

    print("what each county actually is (hidden from the searcher)\n")
    for name, spec in COUNTIES.items():
        print(f"  {name:14s} {spec['note']}")
    print()

    results = {}
    fitted_at_max = {}

    for name, spec in COUNTIES.items():
        grid, scen = grids[name], suites[name]
        results[name] = {}
        print(f"{name}   ({len(scen) - max(TRAIN_SIZES)} held-out cases)")
        print(f"  {'cases':>6}  {'published':>10}  {'home':>10}   learned")
        for n_train in TRAIN_SIZES:
            train, tests = cases_and_tests(grid, scen, n_train)
            # Hold the test set fixed across training sizes so the comparison
            # is not also a comparison of which cases happened to be tested.
            tests = tests[-(len(scen) - max(TRAIN_SIZES)):]
            model = fit_county(name, train, grid)
            base = float(np.mean([fraction_swept(p, s.true_rc) for s, p in tests]))
            home = float(np.mean([
                fraction_swept(model.adjust(p, grid, s.true_anchor_rc), s.true_rc)
                for s, p in tests]))
            results[name][n_train] = {"published": base, "home": home,
                                      "weights": model.weights.tolist()}
            if n_train == max(TRAIN_SIZES):
                fitted_at_max[name] = model
            learned = ", ".join(f"{w:+.2f}" for w in model.weights)
            print(f"  {n_train:>6}  {100 * base:9.1f}%  {100 * home:9.1f}%   [{learned}]")
        print()

    print("is what it learned local? (32 cases, held-out)\n")
    print(f"  {'county':14s} {'published':>10} {'home':>10} {'transplant':>11}")
    for name in COUNTIES:
        grid, scen = grids[name], suites[name]
        _, tests = cases_and_tests(grid, scen, max(TRAIN_SIZES))
        base = float(np.mean([fraction_swept(p, s.true_rc) for s, p in tests]))
        home = float(np.mean([
            fraction_swept(fitted_at_max[name].adjust(p, grid, s.true_anchor_rc), s.true_rc)
            for s, p in tests]))
        others = [o for o in COUNTIES if o != name]
        trans = float(np.mean([
            np.mean([fraction_swept(fitted_at_max[o].adjust(p, grid, s.true_anchor_rc),
                                    s.true_rc) for s, p in tests])
            for o in others]))
        results[name]["transplant"] = trans
        print(f"  {name:14s} {100 * base:9.1f}% {100 * home:9.1f}% {100 * trans:10.1f}%")

    print("\n\nflying it: the same drone and the same code, on held-out cases\n")
    print(f"  {'county':14s} {'cold':>18} {'experienced':>18} {'neighbour':>18}")
    end_to_end = {}
    for name, spec in COUNTIES.items():
        grid, scen = grids[name], suites[name]
        _, tests = cases_and_tests(grid, scen, max(TRAIN_SIZES))
        pod = pod_field(grid)
        others = [o for o in COUNTIES if o != name]
        arms = {"cold": None, "experienced": fitted_at_max[name],
                "neighbour": fitted_at_max[others[0]]}
        row = {}
        for arm, model in arms.items():
            found, periods = 0, []
            for s_, _p in tests:
                for rep in range(3):
                    r = run_scenario(grid, pod, s_, "none", CFG,
                                     np.random.default_rng(stable_seed(s_.id, rep)),
                                     county=model)
                    found += bool(r.found)
                    periods.append(r.periods_to_find if r.found else CFG.max_periods)
            n = len(tests) * 3
            row[arm] = {"find_rate": found / n, "mean_periods": float(np.mean(periods))}
            print(f"    {arm:>12}  {100 * found / n:5.1f}% found, "
                  f"{np.mean(periods):4.1f} periods", end="")
            if arm == "neighbour":
                print()
        print(f"  {name}")
        end_to_end[name] = row

    out = ROOT / "runs" / "county_learning.json"
    out.write_text(json.dumps(
        {"counties": {k: {kk: v[kk] for kk in v} for k, v in results.items()},
         "end_to_end": end_to_end,
         "train_sizes": TRAIN_SIZES, "n_total": n_total}, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
