"""Print the main tables of the write-up from the experiment outputs (run the scripts in README.md first).

    python experiments/make_tables.py            # all sections whose inputs exist
    python experiments/make_tables.py --only bidding operator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genco_market.core.config import REPO, scratch_dir  # noqa: E402

RES = scratch_dir("results")
EXP2_TAGS = ["heatv4", "heatv4_hs", "ext2"]  # 40 random + 30 heatwave/thunderstorm + 40 extension-event states
SCENARIO = {"none": "No event", "heatwave": "Heatwave", "thunderstorm": "Thunderstorm", "planned_outage": "Planned outage",
            "gas_price": "Gas-price spike", "gas_price+heatwave": "Gas-price spike + heatwave",
            "planned_outage+thunderstorm": "Planned outage + thunderstorm", "cold_snap": "Cold snap",
            "wildfire_psps": "Wildfire shutoff", "high_wind": "High wind", "solar_drop": "Solar drop",
            "demand_response": "Conservation call", "market_power_review": "Market-power review",
            "geomagnetic_storm": "Geomagnetic storm", "cyber_physical": "Substation threat"}
METHOD = {"ac": "AC-OPF", "dc": "DC-OPF", "mf_dc": "AC + DC", "genco": "Fine-tuned GENCO"}


def md(df: pd.DataFrame, fmt) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [v if isinstance(v, str) else fmt(v) for v in r]) + " |")
    return "\n".join(lines)


def missing(*paths: Path) -> bool:
    gone = [p for p in paths if not p.exists()]
    for p in gone:
        print(f"  (missing {p}; skipped)")
    return bool(gone)


def price_head():
    print("\n## Section 3: price head (Exp 1)")
    p = REPO / "results" / "exp1_summary.md"
    if not missing(p):
        print(p.read_text())


def bidding():
    print("\n## Section 4: one strategic firm, true scenario distribution (110 states)")
    fs = [RES / f"exp2_regret_{t}_ft2.parquet" for t in EXP2_TAGS]
    if missing(*fs):
        return
    df = pd.concat([pd.read_parquet(f).assign(src=t) for f, t in zip(fs, EXP2_TAGS)])
    t = df.groupby(["method", "budget"]).regret_pct.mean().unstack("budget").loc[list(METHOD), [0.5, 1.0, 2.0, 5.0]]
    t.index, t.index.name = [METHOD[m] for m in t.index], "Estimator"
    t.columns = [f"{c:g} s" for c in t.columns]
    print(md(t, lambda v: f"{v:.2f}%"))

    print("\n### By scenario type, 0.5 s budget")
    x = df[df.budget == 0.5]
    t = x.groupby(["events", "method"]).regret_pct.mean().unstack("method")[list(METHOD)]
    n = x[["events", "src", "state"]].drop_duplicates().groupby("events").size()
    t.insert(0, "States", n.astype(str))
    t = t.loc[[e for e in SCENARIO if e in t.index]]
    t.index, t.index.name = [SCENARIO[e] for e in t.index], "Scenario"
    t.columns = ["States"] + [METHOD[m] for m in METHOD]
    print(md(t, lambda v: f"{v:.2f}%"))


def end_to_end():
    print("\n## Section 4: end to end from L3 bulletins, 0.5 s budget (240 decisions, first one dropped)")
    fs = [RES / f"episodes_e2e_ft2_b05_{o}.parquet" for o in (0, 20, 40)]
    if missing(*fs):
        return
    d = pd.concat([pd.read_parquet(f) for f in fs])
    d = d[~((d.episode == 0) & (d.t == 0))]  # solver-pool cold start: AC finishes nothing on the very first call
    d["hit"] = (d.markup - d.oracle_markup).abs() < 1e-9
    g = d.groupby("agent").agg(n=("regret", "size"), regret=("regret", "mean"), opt=("oracle_profit", "mean"), hit=("hit", "mean"))
    names = {"llm/ac": "LLM + AC-OPF", "llm/mf_dc": "LLM + AC + DC", "llm/mf_genco": "LLM + AC + fine-tuned GENCO",
             "llm/genco": "LLM + fine-tuned GENCO"}
    t = pd.DataFrame({"Regret (% of optimal profit)": (100 * g.regret / g.opt).map("{:.2f}%".format),
                      "Optimal bid selected": (100 * g.hit).map("{:.0f}%".format),
                      "Regret ($/h)": g.regret.map("{:,.0f}".format), "Decisions": g.n.astype(str)})
    t = t.loc[[a for a in names if a in g.index]]
    t.index, t.index.name = [names[a] for a in t.index], "Method"
    print(md(t, str))
    p = d.pivot_table(index=["episode", "t"], columns="agent", values="regret")
    for other in ("llm/ac", "llm/mf_dc"):
        if {"llm/genco", other} <= set(p.columns):
            x = (p["llm/genco"] - p[other]).dropna()
            print(f"  paired: LLM + fine-tuned GENCO - {names[other]} = {x.mean():+.0f} ± {1.96 * x.sem():.0f} $/h (95%)")


def three_firms():
    print("\n## Section 5: three strategic firms, rivals' bids unobserved (35 normal states x 20 repetitions)")
    p = REPO / "results" / "l3abd_fid_nash_normal_nash_gap.csv"
    if missing(p):
        return
    t = pd.read_csv(p, index_col="method").loc[["ac", "dc", "genco", "mf_genco"], ["2", "5", "10"]]
    t.index = ["AC-OPF", "DC-OPF", "GENCO", "AC + GENCO"]
    t.index.name, t.columns = "Method", ["2 s", "5 s", "10 s"]
    print(md(t, lambda v: "N/A" if pd.isna(v) else f"${v:,.0f}"))


def regulation():
    print("\n## Section 6: regulation, firms anticipate the rule (share of the markup premium removed)")
    ps = {lab: REPO / "results" / f"l3abd_rules_{lab}.csv" for lab in ("normal", "scarcity")}
    if missing(*ps.values()):
        return
    rules = {"R4_A_only": "Targeted mitigation (Firm A only)", "R1_cap2": "Markup cap 2x", "R1_cap1.5": "Markup cap 1.5x",
             "R2_c0.1_i25": "Conduct-and-impact, theta = $25/MWh", "R2_c0.1_i2": "Conduct-and-impact, theta = $2/MWh",
             "R3_shed": "Pivotal-supplier test", "R5_price300": "Price cap $300/MWh"}
    cols = {}
    for lab, p in ps.items():
        d = pd.read_csv(p)
        d = d[d.behaviour == "anticipating"].set_index("rule")
        cols[lab.capitalize()] = d.premium_removed
        if lab == "scarcity":
            cols["Scarcity premium removed"] = d.scarcity_removed_frac
    t = pd.DataFrame(cols).loc[list(rules)]
    t.index, t.index.name = [rules[r] for r in t.index], "Mechanism"
    print(md(t, lambda v: f"{100 * v:.0f}%"))
    d = pd.read_csv(ps["normal"]).set_index(["behaviour", "rule"])
    print(f"  markup cap 2x with bids held fixed (naive firms): {100 * d.loc[('naive', 'R1_cap2'), 'premium_removed']:.0f}% removed")


def operator():
    print("\n## Section 7: outage screening, true distribution (200 requests x 100 repetitions)")
    p = RES / "l4_e42_op_200.parquet"
    if missing(p):
        return
    d = pd.read_parquet(p)
    d = d[d.deferral_cost == 4000]
    names = {"genco": "Operator-adapted GENCO", "mf_dc": "AC + DC", "ac": "AC-OPF", "dc": "DC-OPF"}
    t = d.groupby(["method", "budget"]).regret.mean().unstack("budget").loc[list(names), [0.5, 1.0, 2.0]]
    t.index, t.index.name = [names[m] for m in t.index], "Estimator"
    t.columns = [f"{c:g} s" for c in t.columns]
    print(md(t, lambda v: f"${v:,.0f}"))
    e = d.groupby(["budget", "method", "episode"]).regret.mean().unstack("method")
    for b in (0.5, 1.0, 2.0):
        x = e.loc[b, "genco"] - e.loc[b, "mf_dc"]
        print(f"  paired at {b:g} s: operator GENCO - AC + DC = {x.mean():+.0f} ± {x.sem():.0f} $/request (1 s.e.)")
    from l4_offline import load_all

    from genco_market.env.outage import OutageRequest, option_costs
    from genco_market.eval.outage_oracle import oracle_delta

    reg = []
    for _, z in load_all():
        req = OutageRequest(**z["meta"]["request"])
        delta, _, _ = oracle_delta(z)
        oc = option_costs(delta, req, 4000)
        reg.append(oc[req.options(len(delta)).index(req.s_req)] - oc.min())
    print(f"  approve every request as filed: {np.mean(reg):,.0f} $/request ({len(reg)} requests)")


SECTIONS = {"price_head": price_head, "bidding": bidding, "end_to_end": end_to_end, "three_firms": three_firms,
            "regulation": regulation, "operator": operator}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SECTIONS), default=list(SECTIONS))
    for name in ap.parse_args().only:
        SECTIONS[name]()
