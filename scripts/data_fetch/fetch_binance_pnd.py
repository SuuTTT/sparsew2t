#!/usr/bin/env python3
"""Hourly SYM/BTC klines from Binance Vision (public archive) for every coin pumped on Binance in the La Morgia et al.
Telegram pump table, plus the anchor verification sheet built from the pump table (pump hour = peak = publication).
Output: <out>/bars_1h.csv (date,ticker,open,high,low,close,volume,amount) and <out>/pnd_sheet.csv"""
import argparse, io, os, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (academic research crawler)"}
URL = "https://data.binance.vision/data/spot/monthly/klines/{s}BTC/1h/{s}BTC-1h-{y}-{m:02d}.zip"


def month_frame(sym, y, m):
    for k in range(3):
        try:
            r = requests.get(URL.format(s=sym, y=y, m=m), headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]), header=None).iloc[:, :8]
                df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount"]
                return df
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pumps", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2017-07"); ap.add_argument("--end", default="2021-03"); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    p = pd.read_csv(a.pumps); p = p[p["exchange"].str.lower() == "binance"].copy()
    p["ts"] = pd.to_datetime(p["date"] + " " + p["hour"].astype(str), errors="coerce").dt.floor("h")
    syms = sorted(p["symbol"].astype(str).str.upper().unique())
    months = pd.period_range(a.start, a.end, freq="M")
    tasks = [(s, mo.year, mo.month) for s in syms for mo in months]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda t: (t[0], month_frame(*t)), tasks))
    frames = []
    for s, df in res:
        if df is not None and len(df):
            df = df.copy(); df["ticker"] = s
            unit = "us" if df["open_time"].max() > 1e14 else "ms"
            df["date"] = pd.to_datetime(df["open_time"], unit=unit)
            frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume", "amount"]])
    bars = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    bars.to_csv(os.path.join(a.out, "bars_1h.csv"), index=False)
    have = set(bars["ticker"].unique())
    sh = p[p["symbol"].str.upper().isin(have)].dropna(subset=["ts"]).reset_index(drop=True)
    sheet = pd.DataFrame({"case_id": [f"PND-{i:04d}" for i in range(len(sh))], "source": "TELEGRAM", "ticker": sh["symbol"].str.upper(),
                          "anomaly_class": "pump_dump", "onset_date": sh["ts"].astype(str), "end_date": (sh["ts"] + pd.Timedelta(hours=1)).astype(str),
                          "peak_date": sh["ts"].astype(str), "release_date": sh["ts"].astype(str), "date_resolution": "hour",
                          "url": "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset", "verifier": "La Morgia et al. 2020", "notes": sh["group"]})
    sheet.to_csv(os.path.join(a.out, "pnd_sheet.csv"), index=False)
    print("coins", len(have), "of", len(syms), "| bars", len(bars), "| pump cases", len(sheet), "of", len(p))


if __name__ == "__main__":
    main()
