#!/usr/bin/env python3
"""Paired case-bootstrap comparison of two runs on identical folds / seeds / budgets (e.g. ours vs Deep SAD).

    python3 scripts/compare_runs.py runs/sparsewarn2trade/<ts> runs/baseline_deep_sad/<ts> --metric event_AUROC
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warn2trade.backtest.event_eval import paired_event_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--metric", default="event_AUROC")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    a, b = pd.read_csv(os.path.join(args.run_a, "scores.csv")), pd.read_csv(os.path.join(args.run_b, "scores.csv"))
    out = []
    for (protocol, K), ga in a.groupby(["protocol", "K"]):
        gb = b[(b.protocol == protocol) & (b.K == K)]
        if gb.empty:
            continue
        key = ["fold", "seed", "t", "asset"]
        if not ga[key].sort_values(key).reset_index(drop=True).equals(gb[key].sort_values(key).reset_index(drop=True)):
            print(f"WARNING protocol={protocol} K={K}: runs are not on identical test rows; comparison skipped")
            continue
        out.append({"protocol": protocol, "K": K, **paired_event_delta(ga, gb, args.metric, args.n_boot)})
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
