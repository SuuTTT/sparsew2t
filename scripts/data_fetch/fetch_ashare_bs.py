#!/usr/bin/env python3
"""A-share daily bars from Baostock (free API): delisting-inclusive universe (union of the stock list at every year end),
unadjusted OHLCV + amount, suspension (tradestatus) and ST flags, back-adjustment factor. Parallel sessions, resumable.
Output: <out>/daily_bars.csv: date,ticker,open,high,low,close,volume,amount,adj_factor,suspended,is_st ; <out>/universe.csv"""
import argparse, os, sys, time
from multiprocessing import Pool
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pylib"))
import baostock as bs

FIELDS = "date,code,open,high,low,close,volume,amount,tradestatus,isST"
A_PREFIX = ("sh.60", "sh.68", "sz.00", "sz.30")


class APIError(Exception):
    pass


def rows(rs):
    if rs.error_code != "0":
        raise APIError(rs.error_msg)
    out = []
    while rs.next():
        out.append(rs.get_row_data())
    if rs.error_code != "0":
        raise APIError(rs.error_msg)
    return pd.DataFrame(out, columns=rs.fields)


def init():
    bs.login()


def fetch(args):
    code, out, beg, end = args
    cache = os.path.join(out, "cache", code.replace(".", "_") + ".csv")
    if os.path.exists(cache):
        return code
    k = None
    for attempt in range(6):
        try:
            k = rows(bs.query_history_k_data_plus(code, FIELDS, start_date=beg, end_date=end, frequency="d", adjustflag="3"))
            a = rows(bs.query_adjust_factor(code=code, start_date="1990-01-01", end_date=end))
            break
        except Exception:
            k = None
            time.sleep(min(60, 5 * 2 ** attempt))
            try:
                bs.logout()
            except Exception:
                pass
            bs.login()
    if k is None:
        return None                                                     # failure: no cache file, retried on the next run
    if k is None or k.empty:
        open(cache, "w").write("")
        return code
    for c in ("open", "high", "low", "close", "volume", "amount"):
        k[c] = pd.to_numeric(k[c], errors="coerce")
    k["date"] = pd.to_datetime(k["date"])
    if len(a):
        a["dividOperateDate"] = pd.to_datetime(a["dividOperateDate"])
        a["backAdjustFactor"] = pd.to_numeric(a["backAdjustFactor"], errors="coerce")
        f = a.set_index("dividOperateDate")["backAdjustFactor"].sort_index()
        k["adj_factor"] = f.reindex(k["date"], method="ffill").to_numpy()
        k["adj_factor"] = k["adj_factor"].fillna(f.iloc[0] if len(f) else 1.0)
    else:
        k["adj_factor"] = 1.0
    k["ticker"] = code.split(".")[1]
    k["suspended"] = (k["tradestatus"] != "1").astype(int)
    k["is_st"] = (k["isST"] == "1").astype(int)
    k[["date", "ticker", "open", "high", "low", "close", "volume", "amount", "adj_factor", "suspended", "is_st"]].to_csv(cache, index=False)
    return code


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--beg", default="2008-01-01"); ap.add_argument("--end", default="2026-09-30"); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--fetch-only", action="store_true", help="do not concatenate (second machine)")
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    uni_path = os.path.join(a.out, "universe.csv")
    if os.path.exists(uni_path):
        uni = pd.read_csv(uni_path)
    else:
        bs.login()
        days = [f"{y}-12-31" for y in range(2008, 2026)] + ["2026-09-30"]
        parts = []
        for d in days:
            for back in range(0, 7):                                    # last trading day at or before the date
                df = rows(bs.query_all_stock(day=(pd.Timestamp(d) - pd.Timedelta(days=back)).strftime("%Y-%m-%d")))
                if len(df):
                    parts.append(df); break
        bs.logout()
        uni = pd.concat(parts).drop_duplicates("code")
        uni = uni[uni["code"].str.startswith(A_PREFIX)]
        uni.to_csv(uni_path, index=False)
    codes = uni["code"].tolist()
    order = codes[::-1] if a.reverse else codes
    print("universe", len(codes), flush=True)
    failed = []
    with Pool(a.procs, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(fetch, [(c, a.out, a.beg, a.end) for c in order]), 1):
            if r is None:
                failed.append(i)
            if i % 250 == 0:
                print("fetched", i, "/", len(codes), "failed so far", len(failed), flush=True)
    if failed:
        print("FAILED", len(failed), "codes; rerun the script to retry them", flush=True)
    if a.fetch_only:
        print("fetch-only: done", flush=True)
        return
    parts = []
    for c in codes:
        p = os.path.join(a.out, "cache", c.replace(".", "_") + ".csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            parts.append(pd.read_csv(p, dtype={"ticker": str}))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(a.out, "daily_bars.csv"), index=False)
    pd.DataFrame({"ticker": [c.split(".")[1] for c in codes], "name": uni["code_name"].tolist()}).to_csv(os.path.join(a.out, "universe_names.csv"), index=False)
    print("rows", len(df), "tickers", df["ticker"].nunique())


if __name__ == "__main__":
    main()
