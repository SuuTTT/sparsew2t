#!/usr/bin/env python3
"""Run a rolling SparseWarn2Trade experiment (or a baseline) over label budgets x seeds x protocols.

    python3 scripts/run_experiment.py --config configs/smoke.yaml
    python3 scripts/run_experiment.py --config configs/baselines/deep_sad.yaml --dataset-config configs/datasets/csrc_sanctions.yaml
Outputs under <out_dir>/<name>/<timestamp>/: rows.csv (long, per fold x seed x rule x bps), summary.csv (mean/std),
scores.csv (per test row: score, val threshold, eval label, case id -> pooled event-level analysis), event_level.csv
(case-bootstrap CIs per method x protocol x K), label_efficiency.csv (over cases actually used), cost_curve.csv,
config.yaml, anchors.csv, lf_report.csv.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from _sparse_common import load_cfg, build
from warn2trade.utils import set_seed, get_logger
from warn2trade.data.anchors import anchor_summary
from warn2trade.data.weak_labelers import lf_report
from warn2trade.backtest.walk_forward import RollingExperiment, KEY_METRICS
from warn2trade.backtest.event_eval import event_level_summary
from warn2trade.backtest import metrics as M


def parse_k(vals):
    return [None if str(v).lower() in ("full", "none", "null") else int(v) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-config", default=None, help="dataset overlay (no _base_), e.g. configs/datasets/csrc_sanctions.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--k", nargs="*")
    ap.add_argument("--protocols", nargs="*", choices=["release", "naive", "oracle"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.dataset_config, args.override)
    log = get_logger("warn2trade")
    seeds = args.seeds or cfg["seeds"]
    k_values = parse_k(args.k) if args.k else cfg["label_budget"]["k_values"]
    protocols = args.protocols or cfg["label_budget"].get("protocols", ["release"])
    dev = cfg["experiment"].get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
    set_seed(int(seeds[0]))

    builder, panel, table, lfs = build(cfg)
    log.info("panel T=%d N=%d | anchors:\n%s", panel.T, panel.N, anchor_summary(table).to_string())
    out_dir = os.path.join(args.out or cfg["experiment"].get("out_dir", "runs"), cfg["experiment"]["name"], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    exp = RollingExperiment(panel, table, cfg, lfs, log, device)
    rep = lf_report(exp.votes, [lf.name for lf in lfs], exp.eval_arrays["y"])
    log.info("labeling functions (precision against EVALUATION labels):\n%s", rep.to_string())
    rows, scores = exp.run(seeds, k_values, protocols)
    agg = RollingExperiment.aggregate(rows)

    ev = cfg.get("eval", {})
    ev_rows = []
    for (method, protocol, K), g in scores.groupby(["method", "protocol", "K"]):
        s = event_level_summary(g, int(ev.get("n_boot", 500)), int(ev.get("block", 20)))
        k_eff = rows[(rows.method == method) & (rows.protocol == protocol) & (rows.K == K) & (rows.metric == "K_eff")]["value"].mean()
        ev_rows.append({"method": method, "protocol": protocol, "K": K, "K_eff_mean": k_eff, **s})
    event_df = pd.DataFrame(ev_rows)
    les = []
    for (method, protocol), g in event_df.groupby(["method", "protocol"]):
        for metric in ("event_AUROC", "event_AP", "event_recall"):
            if metric in g:
                les.append({"method": method, "protocol": protocol, "metric": metric,
                            **M.label_efficiency_score(g["K_eff_mean"].to_numpy(), g[metric].to_numpy())})
    les_df = pd.DataFrame(les)
    ks = [k for k in agg["K"].unique()]
    cc = pd.concat([RollingExperiment.cost_curve(agg, m, k) for m in ("net_SR", "ovl_IR") for k in ks], ignore_index=True) if ks else pd.DataFrame()

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    table.to_csv(os.path.join(out_dir, "anchors.csv"), index=False)
    rep.to_csv(os.path.join(out_dir, "lf_report.csv"), index=False)
    rows.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    event_df.to_csv(os.path.join(out_dir, "event_level.csv"), index=False)
    les_df.to_csv(os.path.join(out_dir, "label_efficiency.csv"), index=False)
    cc.to_csv(os.path.join(out_dir, "cost_curve.csv"), index=False)

    sub = agg[(agg["bps"] == exp.default_bps) & (agg["metric"].isin(KEY_METRICS)) & (agg["rule"].isin(["detection", "adapter"]))]
    piv = sub.pivot_table(index=["method", "protocol", "K"], columns="metric", values="mean", dropna=False)
    cols = [c for c in KEY_METRICS if c in piv.columns]
    print("\n=== fold-level means @ %.0f bps (detection + shared adapter rule) ===" % exp.default_bps)
    print(piv[cols].round(3).to_string())
    if len(event_df):
        show = [c for c in ("method", "protocol", "K", "K_eff_mean", "n_events", "event_AUROC", "event_AUROC_lo", "event_AUROC_hi",
                            "event_recall", "event_recall_lo", "event_recall_hi", "FA_per_1k", "lead_median") if c in event_df]
        print("\n=== event level, pooled over folds, case-bootstrap 95% CI ===")
        print(event_df[show].round(3).to_string(index=False))
    print("\nsaved ->", out_dir)
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        json.dump({"n_rows": int(len(rows)), "seeds": seeds, "k_values": [str(k) for k in k_values], "protocols": protocols}, f)


if __name__ == "__main__":
    main()
