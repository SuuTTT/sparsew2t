#!/usr/bin/env python3
"""Go / no-go before spending GPU time: how many regulatory cases can a model see, and how many can a test fold score?

    python3 scripts/power_check.py --config configs/default.yaml --dataset-config configs/datasets/csrc_sanctions.yaml
Prints per fold and protocol: cases published by as-of (per class), cases usable inside the training window, test-window
cases (oracle), pooled test cases, and the minimum detectable paired AUROC difference (80% power, 5% two-sided).
NO-GO if pooled test cases < --min-events (default 30): pool more folds / years, add a second regulator, or report the
dataset as detection-only with wide CIs.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from _sparse_common import load_cfg, build
from warn2trade.data.splits import WalkForwardSplitter
from warn2trade.data.anchors import available_anchors, anchor_summary
from warn2trade.backtest.event_eval import minimum_detectable_auroc_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--min-events", type=int, default=30)
    ap.add_argument("--auc", type=float, default=0.75, help="assumed AUROC for the MDE calculation")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.dataset_config, args.override)
    _, panel, table, _ = build(cfg)
    sp = cfg["splits"]
    folds = list(WalkForwardSplitter(panel.T, sp["train"], sp["val"], sp["test"], sp.get("step"), sp.get("purge", 0), sp.get("embargo", 0),
                                     sp.get("anchored", False), sp.get("max_folds")))
    print("anchor summary (all cases):\n", anchor_summary(table).to_string(), "\n")
    rows = []
    for f in folds:
        lo, as_of, t0, t1 = int(f.train[0]), int(f.train[-1]), int(f.test[0]), int(f.test[-1])
        test_cases = table[(table["onset_t"] - int(cfg["label_budget"].get("lead", 3)) <= t1) & (table["peak_t"].fillna(table["end_t"]) >= t0)]
        for proto in cfg["label_budget"].get("protocols", ["release"]) + (["naive"] if "naive" not in cfg["label_budget"].get("protocols", []) else []):
            av = available_anchors(table, as_of, proto)
            usable = av[(av["end_t"] >= lo) & (av["onset_t"] <= as_of)]
            per_cls = usable.groupby("anomaly_class")["case_id"].nunique().to_dict()
            rows.append({"fold": f.k, "protocol": proto, "published_by_asof": int(av["case_id"].nunique()), "usable_in_train": int(usable["case_id"].nunique()),
                         "usable_per_class": per_cls, "test_cases": int(test_cases["case_id"].nunique())})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    rel = df[df["protocol"] == df["protocol"].iloc[0]]
    pooled = int(rel["test_cases"].sum())
    n_neg = int(sum(len(f.test) for f in folds) * panel.N / max(int(cfg.get("eval", {}).get("block", 20)), 1))
    mde = minimum_detectable_auroc_delta(args.auc, max(pooled, 1), max(n_neg, 1))
    print(f"\npooled test cases = {pooled} (min {args.min_events}); effective negative blocks ~ {n_neg}; "
          f"MDE(paired AUROC, 80% power) at AUROC={args.auc:.2f}: {mde:.3f}")
    print("VERDICT:", "GO" if pooled >= args.min_events else "NO-GO (see docstring for remedies)")


if __name__ == "__main__":
    main()
