#!/usr/bin/env python3
"""A-share daily bars, delisting-inclusive, from Eastmoney's public kline endpoint (raw and back-adjusted).
Universe: the Eastmoney A-share list (includes delisted / PT / ST names) plus any extra codes (e.g. CSRC case codes).
Output: <out>/daily_bars.csv with date,ticker,open,high,low,close,volume,amount,adj_factor (adj = hfq close / raw close).
Polite: `--workers` threads, ~`--rps` requests per second overall, retries with backoff; resumable (per-code cache)."""
import argparse, os, time, threading, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (academic research crawler)"}
LIST = "https://push2.eastmoney.com/api/qt/clist/get?pn={pn}&pz=500&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14"
KL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57"
      "&klt=101&fqt={fqt}&beg={beg}&end={end}")
_lock = threading.Lock(); _last = [0.0]


def throttle(rps):
    with _lock:
        wait = _last[0] + 1.0 / rps - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(url, rps, tries=6):
    for k in range(tries):
        throttle(rps)
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(min(30, 2 ** k))
    return None


def secid(code):
    return ("1." if code.startswith(("6", "9")) else "0.") + code


def universe(rps):
    rows, pn = [], 1
    while True:
        d = get(LIST.format(pn=pn), rps)
        diff = (d or {}).get("data", {}) or {}
        items = diff.get("diff") or {}
        items = items.values() if isinstance(items, dict) else items
        items = list(items)
        if not items:
            break
        rows += [(x["f12"], x["f14"]) for x in items]
        pn += 1
    return pd.DataFrame(rows, columns=["ticker", "name"]).drop_duplicates("ticker")


def fetch_code(code, a):
    cache = os.path.join(a.out, "cache", code + ".csv")
    if os.path.exists(cache):
        return cache
    out = {}
    for fqt in (0, 2):
        d = get(KL.format(secid=secid(code), fqt=fqt, beg=a.beg, end=a.end), a.rps)
        kl = ((d or {}).get("data") or {}).get("klines") or []
        out[fqt] = pd.DataFrame([x.split(",") for x in kl], columns=["date", "open", "close", "high", "low", "volume", "amount"]) if kl else None
    if out[0] is None or out[0].empty:
        open(cache, "w").write("")
        return cache
    raw = out[0]
    for c in ("open", "close", "high", "low", "volume", "amount"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["volume"] = raw["volume"] * 100                                # Eastmoney volume is in lots of 100 shares
    if out[2] is not None and not out[2].empty:
        h = out[2].set_index("date")["close"].astype(float)
        raw["adj_factor"] = (h.reindex(raw["date"]).to_numpy() / raw["close"]).round(8)
    else:
        raw["adj_factor"] = 1.0
    raw.insert(1, "ticker", code)
    raw[["date", "ticker", "open", "high", "low", "close", "volume", "amount", "adj_factor"]].to_csv(cache, index=False)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--beg", default="20080101"); ap.add_argument("--end", default="20260930")
    ap.add_argument("--extra-codes", default=None, help="file with one 6-digit code per line (case codes)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--rps", type=float, default=8.0)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    uni = universe(a.rps)
    if a.extra_codes and os.path.exists(a.extra_codes):
        extra = [l.strip() for l in open(a.extra_codes) if l.strip()]
        uni = pd.concat([uni, pd.DataFrame({"ticker": extra, "name": ""})]).drop_duplicates("ticker")
    uni = uni[uni["ticker"].str.match(r"^(00|30|60|68|8|4)\d+")]
    uni.to_csv(os.path.join(a.out, "universe.csv"), index=False)
    print("universe", len(uni), flush=True)
    done = [0]
    def job(c):
        fetch_code(c, a); done[0] += 1
        if done[0] % 200 == 0:
            print("fetched", done[0], "/", len(uni), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, uni["ticker"].tolist()))
    parts = [pd.read_csv(os.path.join(a.out, "cache", c + ".csv"), dtype={"ticker": str}) for c in uni["ticker"]
             if os.path.getsize(os.path.join(a.out, "cache", c + ".csv")) > 0]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(a.out, "daily_bars.csv"), index=False)
    print("rows", len(df), "tickers", df["ticker"].nunique(), "->", os.path.join(a.out, "daily_bars.csv"))


if __name__ == "__main__":
    main()
