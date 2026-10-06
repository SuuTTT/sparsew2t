#!/usr/bin/env python3
"""Print n random auto-extracted CSRC cases with the decision text around the extracted stock and date range."""
import json, re, sys
import pandas as pd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
sh = pd.read_csv("data/csrc/csrc_sheet_auto.csv", dtype={"ticker": str})
uni = pd.read_csv("data/ashare_bs/universe.csv", dtype=str)
name = dict(zip(uni["code"].str.split(".").str[1], uni["code_name"]))
docs = {}
for line in open("data/csrc/csrc_decisions.jsonl"):
    d = json.loads(line)
    docs["CSRC-" + str(d["id"])] = d
samp = sh.drop_duplicates("case_id").sample(n, random_state=0)
for _, r in samp.iterrows():
    t = docs[r.case_id]["content"]
    nm = name.get(r.ticker, "?")
    i = t.find(r.ticker) if r.ticker in t else (t.find(nm) if nm in t else -1)
    m = re.search(r"\d{4}年\d{1,2}月(\d{1,2}日)?\s*(至|到)", t)
    print("##", r.case_id, r.anomaly_class, "| ticker", r.ticker, nm, "| window", r.onset_date, "->", r.end_date, "| pub", str(r.release_date)[:10])
    print("   stock:", t[max(0, i - 40):i + 40].replace("\n", " ") if i >= 0 else "NOT FOUND")
    print("   date :", t[max(0, m.start() - 30):m.start() + 70].replace("\n", " ") if m else "-")
