#!/usr/bin/env python3
"""Pool a sweep (runs/<tag>/<config>/seed*/.../) into paper tables.

Writes results/<tag>/: rows_all.csv, summary.csv (fold means/std per method x rule x protocol x K x bps x metric),
event_level.csv (pooled over folds AND seeds, case-bootstrap CIs), paired.csv (reference method vs every other method,
same cases, paired bootstrap on event AUROC and event recall), dsr.csv, tables.md."""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warn2trade.backtest.event_eval import event_level_summary, paired_event_delta
from warn2trade.backtest.portfolio_eval import deflated_sharpe_from_moments


NEG_FRAC = 1.0


def load(tag, root="runs"):
    """Run directories: <root>/<tag>/<config>/seed*/ (cq layout: rows.csv + scores.csv + DONE at the top) or the older
    hand-run layout <root>/<tag>/<config>/seed*/<name>/<timestamp>/. Several tags may be given comma-separated."""
    rows, scores = [], []
    dirs = []
    for t in tag.split(","):
        dirs += [d for d in glob.glob(os.path.join(root, t, "*", "seed*")) if os.path.exists(os.path.join(d, "DONE"))]
        dirs += [d for d in glob.glob(os.path.join(root, t, "*", "seed*", "*", "*")) if os.path.exists(os.path.join(d, "DONE"))]
    for d in dirs:
        if not os.path.exists(os.path.join(d, "rows.csv")):
            continue
        cfg = re.sub(r"_K[^_]+$", "", os.path.relpath(d, root).split(os.sep)[1])   # one-job-per-K folders: default_K3 -> default
        r = pd.read_csv(os.path.join(d, "rows.csv")); r["config"] = cfg; rows.append(r)
        s = pd.read_csv(os.path.join(d, "scores.csv")); s["config"] = cfg
        if "w" not in s:                                                         # runs that saved every row
            s["pct"] = s.groupby(["fold", "seed"])["score"].rank(pct=True)      # on the full test set
            keep = (s["y"] == 1) | (np.random.default_rng(len(s)).random(len(s)) < NEG_FRAC)
            s = s[keep].copy(); s["w"] = np.where(s["y"] == 1, 1.0, 1.0 / NEG_FRAC)
        scores.append(s)
    return pd.concat(rows, ignore_index=True), pd.concat(scores, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ref", default="default")
    ap.add_argument("--root", default="runs"); ap.add_argument("--out", default=None)
    ap.add_argument("--neg-frac", type=float, default=1.0, help="keep this share of negative rows (weighted 1/frac)")
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--bps", type=float, default=5.0)
    a = ap.parse_args()
    out = a.out or os.path.join("results", a.tag); os.makedirs(out, exist_ok=True)
    global NEG_FRAC
    NEG_FRAC = a.neg_frac
    rows, scores = load(a.tag, a.root)
    print("loaded", rows["config"].nunique(), "configs,", len(rows), "rows,", len(scores), "score rows", flush=True)
    rows.to_csv(os.path.join(out, "rows_all.csv"), index=False)
    g = rows.groupby(["config", "rule", "protocol", "K", "bps", "metric"])["value"]
    summ = g.agg(["mean", "std", "count"]).reset_index(); summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    ev = []
    for (cfg, proto, K), gs in scores.groupby(["config", "protocol", "K"]):
        k_eff = rows[(rows.config == cfg) & (rows.protocol == proto) & (rows.K == K) & (rows.metric == "K_eff")]["value"].mean()
        ev.append({"config": cfg, "protocol": proto, "K": K, "K_eff": k_eff, "n_seeds": gs["seed"].nunique(), **event_level_summary(gs, a.n_boot)})
    ev = pd.DataFrame(ev); ev.to_csv(os.path.join(out, "event_level.csv"), index=False)
    pr = []
    ref = scores[scores.config == a.ref]
    for (cfg, proto, K), gb in scores[scores.config != a.ref].groupby(["config", "protocol", "K"]):
        ga = ref[(ref.protocol == proto) & (ref.K == K)]
        common = set(ga["seed"]) & set(gb["seed"])
        if ga.empty or not common:
            continue
        ga, gb = ga[ga.seed.isin(common)], gb[gb.seed.isin(common)]
        for m in ("event_AUROC", "event_recall"):
            pr.append({"ref": a.ref, "other": cfg, "protocol": proto, "K": K, **paired_event_delta(ga, gb, m, a.n_boot)})
    pr = pd.DataFrame(pr); pr.to_csv(os.path.join(out, "paired.csv"), index=False)
    m = rows[(rows.bps == a.bps) & rows.metric.isin(["net_sr_bar", "net_n_bars", "net_skew", "net_kurt", "ovl_active_sr_bar", "ovl_active_n_bars", "ovl_active_skew", "ovl_active_kurt"])]
    key = ["config", "rule", "protocol", "K", "fold", "seed"]
    w = m.pivot_table(index=key, columns="metric", values="value").reset_index()
    for pre in ("net", "ovl_active"):
        if f"{pre}_sr_bar" in w:
            tr = w[f"{pre}_sr_bar"].to_numpy()
            w[f"{pre}_DSR_sweep"] = [deflated_sharpe_from_moments(r[f"{pre}_sr_bar"], r[f"{pre}_n_bars"], r[f"{pre}_skew"], r[f"{pre}_kurt"], tr) for _, r in w.iterrows()]
    w.to_csv(os.path.join(out, "dsr.csv"), index=False)
    with open(os.path.join(out, "tables.md"), "w") as f:
        cols = [c for c in ["config", "protocol", "K", "K_eff", "n_events", "n_seeds", "event_AUROC", "event_AUROC_lo", "event_AUROC_hi", "event_AP",
                            "event_recall", "event_recall_lo", "event_recall_hi", "FA_per_1k", "lead_median"] if c in ev]
        f.write("## Event level (pooled over folds and seeds, case bootstrap 95% CI)\n\n" + ev[cols].round(3).to_markdown(index=False) + "\n\n")
        if len(pr):
            f.write("## Paired differences vs " + a.ref + "\n\n" + pr.round(4).to_markdown(index=False) + "\n\n")
        keym = ["AP", "AUROC", "F1_val", "net_SR", "net_SR_perm_p", "ovl_IR", "ovl_IR_perm_p", "ovl_active_ARR", "PLCA_pos"]
        s2 = summ[(summ.bps == a.bps) & summ.metric.isin(keym) & summ.rule.isin(["detection", "adapter"])]
        piv = s2.pivot_table(index=["config", "protocol", "K"], columns="metric", values="mean")
        f.write(f"## Fold means @ {a.bps:g} bps (detection + shared adapter rule)\n\n" + piv.round(3).to_markdown() + "\n")
    print(open(os.path.join(out, "tables.md")).read()[:6000])


if __name__ == "__main__":
    main()
