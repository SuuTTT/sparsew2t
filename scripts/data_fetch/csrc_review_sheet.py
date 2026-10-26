#!/usr/bin/env python3
"""Auto-extracted CSRC anchor sheet -> human review sheet for independent double verification.

One row per (case, stock), with the extracted fields, the decision text around the stock, every date range in the text and
the class keyword, plus blank reviewer columns. Rows whose anchor falls in a walk-forward test fold come first (they decide
the reported results). Outputs in --out:
  review_sheet.csv      the sheet (UTF-8 with BOM so Excel / WPS show Chinese); each reviewer fills an own copy
  decisions_fulltext.txt  full decision text per case, for anything the snippets do not settle
  REVIEW_GUIDE.md       instructions for the reviewers
"""
import argparse, json, os, re, sys
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_csrc import CLS, D_RANGE  # noqa: E402  same patterns the extractor used

REVIEW_COLS = ["stock_ok", "window_ok", "class_ok", "release_ok", "fix_ticker", "fix_onset", "fix_end", "fix_class",
               "fix_release", "reviewer", "comment"]

GUIDE = """# CSRC case review guide

Each reviewer fills an own copy of `review_sheet.csv` (rename it `review_<initials>.csv`) without looking at the other copy.
One row = one (decision, stock) pair extracted automatically from the CSRC penalty decision at `url`.

For every row fill the four checks with `Y` (correct), `N` (wrong) or `?` (cannot tell from the decision):

| column | question | if `N`, fill |
|---|---|---|
| stock_ok | Is `ticker` / `stock_name` a stock the penalised party actually traded in the misconduct? (not the issuer of a disclosure case, not an account or fund name) | `fix_ticker` (6 digits) or `NONE` if no traded stock |
| window_ok | Do `onset_date`..`end_date` match the trading period the decision gives for THIS stock (to within a few days)? | `fix_onset`, `fix_end` (YYYY-MM-DD) |
| class_ok | Is `anomaly_class` right? manipulation = 操纵 (incl. 虚假申报, 连续买卖, 对倒, 蛊惑); insider = 内幕交易; front_running = 利用未公开信息 | `fix_class` (`manipulation`, `insider`, `front_running`, or `other` if not a trading case) |
| release_ok | Is `release_date` the publication date of the decision? | `fix_release` |

Columns to read: `stock_context` (text around the stock), `date_ranges` (every date range in the decision; `>>` marks the one
nearest the stock), `class_context`, `other_stocks_in_case`. If they do not settle it, open `url` or search
`decisions_fulltext.txt` for the `case_id`. Put your initials in `reviewer`; anything else in `comment`.
Priority: rows with `eval_role = test` first (they are scored); `train_only` rows only label training.
Do not edit the extracted columns themselves; corrections go only into the `fix_*` columns.
"""


def ctx(t, i, n, r):
    return t[max(0, i - r):i + n + r].replace("\n", " ")


def stock_context(t, keys, r=60, max_hits=2):
    hits = sorted({m.start(): k for k in keys if k for m in re.finditer(re.escape(k), t)}.items())
    return " ‖ ".join(ctx(t, i, len(k), r) for i, k in hits[:max_hits]) if hits else "NOT FOUND"


def date_ranges(t, keys, r=30, max_hits=8):
    pos = [m.start() for k in keys if k for m in re.finditer(re.escape(k), t)]
    found = list(D_RANGE.finditer(t))
    if not found:
        return "-"
    near = min(found, key=lambda g: min((abs(g.start() - p) for p in pos), default=10 ** 9)).start() if pos else -1
    return " ‖ ".join((">>" if g.start() == near else "") + ctx(t, g.start(), len(g.group(0)), r) for g in found[:max_hits])


def class_context(t, r=40):
    for c, kws in CLS:
        for k in kws:
            i = t.find(k)
            if i >= 0:
                return f"[{k}] " + ctx(t, i, len(k), r)
    return "-"


def first_test_bar(cfg_path):
    s = yaml.safe_load(open(cfg_path))["splits"]
    return s["train"] + s.get("purge", 0) + s["val"] + s.get("purge", 0)    # te_lo of fold 0 (warn2trade/data/splits.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="csrc_sheet_auto_tight.csv (or _all)")
    ap.add_argument("--decisions", required=True, help="csrc_decisions.jsonl")
    ap.add_argument("--universe", required=True, help="Baostock universe.csv (code, code_name)")
    ap.add_argument("--anchors", default=None, help="csrc_anchors_tight.csv: marks which rows reached the panel and their fold role")
    ap.add_argument("--dataset-config", default="configs/datasets/csrc_pilot.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sh = pd.read_csv(a.sheet, dtype={"ticker": str, "case_id": str})
    uni = pd.read_csv(a.universe, dtype=str)
    name = dict(zip(uni["code"].str.split(".").str[1], uni["code_name"]))
    want = set(sh["case_id"])
    docs = {}
    for line in open(a.decisions):
        d = json.loads(line)
        cid = f"CSRC-{d['id']}"
        if cid in want:
            docs[cid] = d

    role = {}
    if a.anchors:
        anc = pd.read_csv(a.anchors, dtype={"asset": str, "case_id": str})
        te = first_test_bar(a.dataset_config)
        role = {(c, s): ("test" if o >= te else "train_only") for c, s, o in zip(anc["case_id"], anc["asset"], anc["onset_t"])}

    rows = []
    for _, r in sh.iterrows():
        d = docs.get(r.case_id, {})
        t = (d.get("content") or "").replace("　", "")
        nm = name.get(r.ticker, "")
        keys = [r.ticker, nm, nm.replace("ST", "").replace("*", "")]
        others = sorted(set(sh.loc[sh.case_id == r.case_id, "ticker"]) - {r.ticker})
        rows.append({
            "eval_role": role.get((r.case_id, r.ticker), "not_in_panel") if role else "unknown",
            "case_id": r.case_id, "ticker": r.ticker, "stock_name": nm, "anomaly_class": r.anomaly_class,
            "onset_date": r.onset_date, "end_date": r.end_date, "release_date": str(r.release_date)[:10],
            "date_resolution": r.date_resolution, "url": r.url, "title": d.get("title", ""),
            "stock_context": stock_context(t, keys), "date_ranges": date_ranges(t, keys), "class_context": class_context(t),
            "other_stocks_in_case": " ".join(f"{o}:{name.get(o, '?')}" for o in others),
            **{c: "" for c in REVIEW_COLS}})
    out = pd.DataFrame(rows)
    order = {"test": 0, "train_only": 1, "unknown": 2, "not_in_panel": 3}
    out = out.sort_values(["eval_role", "case_id", "ticker"], key=lambda s: s.map(order) if s.name == "eval_role" else s)
    out.to_csv(os.path.join(a.out, "review_sheet.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(a.out, "decisions_fulltext.txt"), "w") as f:
        for cid in out["case_id"].drop_duplicates():
            d = docs.get(cid, {})
            f.write(f"==================== {cid} | {d.get('published', '')} | {d.get('title', '')}\n{d.get('url', '')}\n\n"
                    f"{(d.get('content') or 'TEXT MISSING').strip()}\n\n")
    open(os.path.join(a.out, "REVIEW_GUIDE.md"), "w").write(GUIDE)

    print("rows", len(out), "cases", out["case_id"].nunique(), "| missing decision text", sum(c not in docs for c in want))
    print("by role (rows):", out["eval_role"].value_counts().to_dict())
    print("by role (cases):", out.groupby("eval_role")["case_id"].nunique().to_dict())
    print("stock not found in text:", int((out["stock_context"] == "NOT FOUND").sum()))


if __name__ == "__main__":
    main()
