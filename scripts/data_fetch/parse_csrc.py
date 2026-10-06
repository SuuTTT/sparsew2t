#!/usr/bin/env python3
"""CSRC decision text -> AUTO-EXTRACTED anchor sheet (pilot; every row must later be human-verified).

Per decision: class from keywords (manipulation: 操纵证券市场 / 操纵 / 虚假申报 / 连续买卖 / 对倒; insider: 内幕交易; front_running:
利用未公开信息), stock codes (explicit 6-digit codes in stock context) and stock names in quotes matched against the Eastmoney
universe (current names; renamed stocks are missed and counted), conduct window = earliest start / latest end of the date
ranges in the text (day resolution; month-only ranges expanded to month bounds), release = publication time.
Output: <out>/csrc_sheet_auto.csv in the template schema (verifier = auto-regex) + <out>/csrc_parse_report.csv"""
import argparse, json, re, calendar
import pandas as pd

CLS = [("manipulation", ["操纵证券市场", "操纵期货市场", "操纵市场", "虚假申报", "连续买卖", "对倒", "蛊惑"]),
       ("insider", ["内幕交易"]), ("front_running", ["利用未公开信息"])]
D_RANGE = re.compile(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?\s*(?:至|到|—|-|～|~)\s*(?:(\d{4})年)?(\d{1,2})月(?:(\d{1,2})日)?")
D_ONE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
CODE_CTX = re.compile(r"(?:代码|证券代码|股票代码|简称)[^0-9]{0,12}?([03689]\d{5})|[“\"(（]([03689]\d{5})[)）”\"]|([03689]\d{5})\s*[)）]")
QUOTED = re.compile(r"[“\"]([^”\"]{2,10})[”\"]")


def classify(t):
    for c, kws in CLS:
        if any(k in t for k in kws):
            return c
    return None


def clamp_day(y, m, d, end):
    m = min(max(m, 1), 12)
    last = calendar.monthrange(y, m)[1]
    return pd.Timestamp(y, m, min(d, last) if d else (last if end else 1))


TRADE_CTX = re.compile(r"(?:买入|卖出|买卖|交易|操纵|拉抬|申报)[^“\"]{0,6}[“\"]([^”\"]{2,10})[”\"]")


def windows_near(t, pub, keys, radius=200):
    """Date ranges within `radius` chars of any mention of the stock (name or code)."""
    spans = [m.start() for k in keys if k for m in re.finditer(re.escape(k), t)]
    if not spans:
        return None, None, "none"
    sub = " ".join(t[max(0, i - radius):i + radius] for i in spans)
    s_, e_, how = windows(sub, pub, singles=False)
    return s_, e_, ("near_" + how) if s_ is not None else "none"


def windows(t, pub, singles=True):
    out = []
    for g in D_RANGE.finditer(t):
        y1, m1, d1, y2, m2, d2 = g.groups()
        try:
            s = clamp_day(int(y1), int(m1), int(d1) if d1 else 0, False)
            e = clamp_day(int(y2 or y1), int(m2), int(d2) if d2 else 0, True)
        except ValueError:
            continue
        if e >= s and s.year >= 2000 and e <= pub:
            out.append((s, e))
    if out:
        return min(s for s, _ in out), max(e for _, e in out), "range"
    if not singles:
        return None, None, "none"
    singles = []
    for g in D_ONE.finditer(t):
        try:
            d = pd.Timestamp(int(g.group(1)), int(g.group(2)), int(g.group(3)))
        except ValueError:
            continue
        if 2000 <= d.year and d <= pub - pd.Timedelta(days=60):
            singles.append(d)
    if singles:
        return min(singles), max(singles), "single_dates"
    return None, None, "none"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--decisions", required=True); ap.add_argument("--universe", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    uni = pd.read_csv(a.universe, dtype={"ticker": str, "code": str})
    if "code" in uni and "ticker" not in uni:                            # Baostock universe: code = sh.600000, code_name
        uni = pd.DataFrame({"ticker": uni["code"].str.split(".").str[1], "name": uni["code_name"]})
    uni["clean"] = uni["name"].fillna("").str.replace(r"\s|\*|ST|PT|退|Ａ|A$", "", regex=True)
    name2code = {n: c for n, c in zip(uni["clean"], uni["ticker"]) if len(n) >= 2}
    codes_ok = set(uni["ticker"])
    rows, rep = [], []
    for line in open(a.decisions):
        r = json.loads(line)
        t = (r.get("content") or "").replace("　", "")
        pub = pd.Timestamp(r["published"][:10])
        cls = classify(t)
        if cls is None:
            rep.append({"id": r["id"], "class": "other", "status": "not_trading_case"}); continue
        codes = {x for g in CODE_CTX.findall(t) for x in g if x}
        names = {q for q in QUOTED.findall(t)}
        by_name = {name2code[n.replace("ST", "").replace("*", "")] for n in names if n.replace("ST", "").replace("*", "") in name2code}
        traded = {n.replace("ST", "").replace("*", "") for n in TRADE_CTX.findall(t)}
        traded_codes = {name2code[n] for n in traded if n in name2code}
        tick = sorted(traded_codes) if traded_codes else sorted((codes & codes_ok) | by_name | (codes - codes_ok))
        code2name = {v: k for k, v in name2code.items()}
        s, e, how = windows(t, pub)
        status = "ok" if tick and s is not None else ("no_stock" if not tick else "no_window")
        rep.append({"id": r["id"], "class": cls, "status": status, "n_stocks": len(tick), "window_from": how})
        if status != "ok":
            continue
        for k, c in enumerate(tick[:5]):                               # multi-stock decisions: first five instruments
            s_c, e_c, how_c = windows_near(t, pub, [code2name.get(c, ""), c])
            if s_c is None:
                s_c, e_c, how_c = s, e, "decision_" + how
            rows.append({"case_id": f"CSRC-{r['id']}", "source": "CSRC", "ticker": c, "anomaly_class": cls,
                         "onset_date": s_c.date(), "end_date": e_c.date(), "peak_date": "", "release_date": r["published"],
                         "date_resolution": how_c, "url": r["url"], "verifier": "auto-regex",
                         "second_verifier": "", "notes": (r.get("memo") or "").strip()[:20]})
    sheet = pd.DataFrame(rows); rep = pd.DataFrame(rep)
    sheet.to_csv(f"{a.out}/csrc_sheet_auto.csv", index=False); rep.to_csv(f"{a.out}/csrc_parse_report.csv", index=False)
    print(rep.groupby(["class", "status"]).size().to_string())
    if len(sheet):
        print("window source:", sheet["date_resolution"].value_counts().to_dict())
        span = (pd.to_datetime(sheet["end_date"]) - pd.to_datetime(sheet["onset_date"])).dt.days
        print("window length days: median", float(span.median()), "p90", float(span.quantile(0.9)))
    print("sheet rows", len(sheet), "cases", sheet["case_id"].nunique() if len(sheet) else 0,
          "| conduct years", (pd.to_datetime(sheet["onset_date"]).dt.year.value_counts().sort_index().to_dict() if len(sheet) else {}))
    codes = sorted(set(sheet["ticker"])) if len(sheet) else []
    open(f"{a.out}/case_codes.txt", "w").write("\n".join(codes))


if __name__ == "__main__":
    main()
