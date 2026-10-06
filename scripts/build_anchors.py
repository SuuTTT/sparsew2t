#!/usr/bin/env python3
"""Verification sheet (dates) -> bar-indexed anchors CSV for a given panel, plus the per-class match report.

    python3 scripts/build_anchors.py --config configs/default.yaml --dataset-config configs/datasets/csrc_sanctions.yaml \
        --sheet data_raw/anchors/csrc_verified.csv --out data_raw/anchors/csrc_sanction_anchors.csv
"""
from __future__ import annotations
import argparse
import pandas as pd

from _sparse_common import load_cfg
from warn2trade.utils import DATASETS
from warn2trade.data.regulatory.sheet import sheet_to_anchors
from warn2trade.data.anchors import anchor_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.dataset_config, [])
    panel = DATASETS.get(cfg["dataset"]["name"])(cfg["dataset"]).build_panel()
    sheet = pd.read_csv(args.sheet, dtype={"ticker": str})
    if "second_verifier" in sheet and sheet["second_verifier"].isna().any():
        print(f"WARNING: {int(sheet['second_verifier'].isna().sum())} rows lack a second verifier")
    table, report = sheet_to_anchors(sheet, panel.times, panel.assets)
    table.to_csv(args.out, index=False)
    report.to_csv(args.out.replace(".csv", "_match_report.csv"), index=False)
    print(report.to_string(index=False))
    print(anchor_summary(table).to_string() if len(table) else "no anchors matched")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
