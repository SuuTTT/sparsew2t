#!/usr/bin/env python3
"""Download every CSRC HQ administrative-penalty decision (channel 行政处罚) with full text -> JSONL.
Polite: 1 request / 1.2 s, generic User-Agent, retries with backoff. Output: <out>/csrc_decisions.jsonl"""
import argparse, json, os, time
import requests

CH = "28de6b87eda140cb93de4dd10d11867d"
URL = "https://www.csrc.gov.cn/searchList/" + CH + "?_isAgg=true&_isJson=true&_pageSize={ps}&_template=index&page={p}"
UA = {"User-Agent": "Mozilla/5.0 (academic research crawler)"}


def get(url, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(url)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--page-size", type=int, default=50)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    first = get(URL.format(ps=a.page_size, p=1))["data"]
    total = int(first["total"]); per = max(1, len(first["results"]))       # the API caps the page size (20)
    pages = (total + per - 1) // per
    a.page_size = per
    path = os.path.join(a.out, "csrc_decisions.jsonl"); n = 0
    with open(path, "w") as f:
        for p in range(1, pages + 1):
            d = first if p == 1 else get(URL.format(ps=a.page_size, p=p))["data"]
            for r in d["results"]:
                f.write(json.dumps({"title": r.get("title"), "memo": (r.get("memo") or "")[:40], "url": "https:" + r.get("url", ""),
                                    "published": r.get("publishedTimeStr"), "id": r.get("manuscriptId"), "content": r.get("content")}, ensure_ascii=False) + "\n")
                n += 1
            print(f"page {p}/{pages} rows {n}", flush=True)
            time.sleep(1.2)
    print("saved", n, "of", total, "->", path)


if __name__ == "__main__":
    main()
