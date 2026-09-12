"""Interactive explorer for the search loop.

A judging table is a conversation, not a screening. People walk up mid-sentence
and want to poke at the thing: show me one where it fails, what happens if the
witness had not named a direction, which account was it actually running on at
period six. A recorded demo cannot answer any of that.

marimo's reactive model suits this exactly -- change a control and everything
downstream recomputes -- and every run is already serialised with per-period
belief fields, hypothesis weights and the nominations that fired.

    marimo edit notebooks/explorer.py      # to explore
    marimo run notebooks/explorer.py       # to hand to someone else
"""
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    return ROOT, json, mo, np


@app.cell
def _(mo):
    mo.md(
        """
        # Search loop — case explorer

        Conventional search optimises *inside* an assumption. When the subject
        never started where the search assumed, no amount of evidence moves the
        belief to them — every hypothesis is anchored at the planning point.

        Pick a run, a case and an operational period. Everything below reacts.
        """
    )
    return


@app.cell
def _(ROOT, json, mo):
    runs = sorted(p.name for p in (ROOT / "runs").glob("experiment*.json"))
    run_pick = mo.ui.dropdown(
        options=runs,
        value="experiment_jev.json" if "experiment_jev.json" in runs else runs[0],
        label="experiment",
    )
    run_pick
    return json, run_pick


@app.cell
def _(ROOT, json, run_pick):
    data = json.loads((ROOT / "runs" / run_pick.value).read_text())
    arms = sorted({r["arm"] for r in data["runs"]})
    return arms, data


@app.cell
def _(arms, data, mo):
    ARM_NAMES = {
        "none": "library only (no revision)",
        "heuristic": "blind relocation (reads nothing)",
        "llm": "language model writes accounts",
        "jev": "System One calibrated distribution",
    }
    preferred = next((a for a in ("jev", "llm", "heuristic", "none") if a in arms),
                     arms[0])
    arm_pick = mo.ui.dropdown(
        options={ARM_NAMES.get(a, a): a for a in arms},
        value=ARM_NAMES.get(preferred, preferred),
        label="arm",
    )
    kind_pick = mo.ui.dropdown(
        options={
            "B — wrong about WHERE they started": "B",
            "A — premise correct (the null)": "A",
            "C — wrong about WHO they are": "C",
        },
        value="B — wrong about WHERE they started",
        label="failure family",
    )
    mo.hstack([arm_pick, kind_pick], justify="start", gap=2)
    return arm_pick, kind_pick


@app.cell
def _(arm_pick, data, kind_pick, mo):
    pool = [r for r in data["runs"]
            if r["arm"] == arm_pick.value and r["scenario_kind"] == kind_pick.value]
    seen, cases = set(), []
    for r in pool:
        if r["scenario_id"] not in seen:
            seen.add(r["scenario_id"])
            cases.append(r)
    labels = {
        f"{r['scenario_id']}  [{r['scenario_subkind']}]  "
        f"{'FOUND p' + str(r['periods_to_find']) if r['found'] else 'not found'}": r
        for r in cases
    }
    first_found = next((k for k in labels if "FOUND" in k), list(labels)[0])
    case_pick = mo.ui.dropdown(options=labels, value=first_found, label="case")
    case_pick
    return case_pick


@app.cell
def _(case_pick, mo):
    run = case_pick.value
    period = mo.ui.slider(
        1, len(run["trace"]), value=min(len(run["trace"]), 1), step=1,
        label="operational period", show_value=True, full_width=True,
    )
    period
    return period, run


@app.cell
def _(mo, period, run):
    t = run["trace"][period.value - 1]
    found = " · **SUBJECT LOCATED**" if t["found"] else ""
    revised = " · **PREMISE FAILING**" if t["trigger_fired"] else ""
    mo.md(
        f"""
        ### Period {t['period']}{revised}{found}

        Leading account: **{t['leader_label']}** — posterior
        {t['leader_weight']:.2f}, **{100 * t['leader_exhaustion']:.0f}% of what it
        predicted has been covered with no contact**.

        Swept {t['track_km']:.0f} km this period. Probability the subject would
        have been found by now if any current account were right:
        {100 * t['cumulative_pos']:.0f}%.
        """
    )
    return (t,)


@app.cell
def _(mo, run):
    # How the leading account and its disconfirmation move across the search.
    _body = "\n".join(
        f"| {x['period']} | {x['leader_label'][:44]} | {x['leader_weight']:.2f} | "
        f"{100 * x['leader_exhaustion']:.0f}% | {100 * x['cumulative_pos']:.0f}% |"
        for x in run["trace"]
    )
    mo.md(
        "#### The premise, period by period\n\n"
        "| period | leading account | posterior | ruled out | P(found by now) |\n"
        "|---|---|---|---|---|\n" + _body
    )
    return


@app.cell
def _(mo, t):
    if t["nominations"]:
        blocks = []
        for n in t["nominations"]:
            cites = "".join(f"\n  - cites: _{c}_" for c in n.get("evidence_cited", [])[:2])
            blocks.append(
                f"**{n['label']}** — start {n['distance_km']:.1f} km on bearing "
                f"{n['bearing_deg']:.0f}°, profile `{n['profile_key']}`, "
                f"prior {n['prior']:.2f}\n\n  {n['narrative']}{cites}"
            )
        out = mo.md(
            f"#### The premise was abandoned here\n\n_{t['trigger_reason']}_\n\n"
            + "\n\n".join(blocks)
        )
    else:
        out = mo.md("")
    out
    return


@app.cell
def _(mo, run):
    mo.md(
        f"""
        ---
        **Outcome.** The subject was **{run['true_distance_km']:.1f} km** from the
        planning point. {'Found in period ' + str(run['periods_to_find']) if run['found']
        else 'Not found within the period budget'}. Revisions: {run['revisions']}.
        Final account: _{run['final_leader']}_ ({run['final_leader_origin']}).

        The searcher never saw the true location. It is shown here only so the
        outcome can be read.
        """
    )
    return


@app.cell
def _(arm_pick, data, mo, np):
    def _rate(arm, kind, key):
        rs = [r for r in data["runs"] if r["arm"] == arm and r["scenario_kind"] == kind]
        if not rs:
            return float("nan")
        if key == "found":
            return 100 * float(np.mean([r["found"] for r in rs]))
        return 100 * float(np.mean([r["periods_to_localize"] is not None for r in rs]))

    _arms_here = sorted({r["arm"] for r in data["runs"]})
    _body = "\n".join(
        f"| {a} | {_rate(a, 'B', 'found'):.0f}% | {_rate(a, 'B', 'loc'):.0f}% | "
        f"{_rate(a, 'A', 'found'):.0f}% |"
        for a in _arms_here
    )
    mo.md(
        "### Across the whole suite\n\n"
        "| arm | type B find | type B localised | type A find (the null) |\n"
        "|---|---|---|---|\n" + _body +
        "\n\n_Type A is the control: revision should not help when the premise "
        "was already right._"
    )
    return


if __name__ == "__main__":
    app.run()
