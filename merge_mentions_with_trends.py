#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser("Join mention data with yearly trends to produce LLM-ready JSONL")
    p.add_argument("--mentions", default="data/cur/clothing_mentions.parquet",
                   help="Input mentions parquet")
    p.add_argument("--trends", default="data/clothing_yearly_trends.csv",
                   help="Yearly trends CSV from clothing_yearly_trends.py")
    p.add_argument("--min-pageviews", type=int, default=100_000,
                   help="Filter: keep mentions from songs with >= pageviews")
    p.add_argument("--out", default="data/cur/lyrics_mentions_enriched.jsonl",
                   help="Output JSONL path")
    return p.parse_args()

def main():
    args = parse_args()
    m = pd.read_parquet(args.mentions)
    t = pd.read_csv(args.trends)

    # normalize inputs
    m["canonical_label"] = m["canonical_label"].astype(str)
    m["year"] = pd.to_datetime(m["release_date"], errors="coerce").dt.year
    m["pageviews"] = pd.to_numeric(m.get("pageviews", 0), errors="coerce").fillna(0).astype(int)
    m["hot"] = m.get("hot", False).astype(bool)

    # filter by popularity
    m = m[m["pageviews"] >= args.min_pageviews].copy()
    if m.empty:
        print("No mentions remain after pageviews filter; nothing to write.")
        return 0

    # trends
    t["label"] = t["label"].astype(str)
    t["year"] = t["year"].astype(int)

    # join
    joined = m.merge(
        t[["label", "year", "yearly_mean"]],
        left_on=["canonical_label", "year"],
        right_on=["label", "year"],
        how="left",
    )
    joined["yearly_mean"] = pd.to_numeric(joined["yearly_mean"], errors="coerce")

    # popularity weight
    joined["popularity_weight"] = np.log1p(joined["pageviews"]).astype(float)

    # minimal training record; you can expand as needed
    fields = [
        "song_id","artist","title","release_date",
        "pageviews","hot","mention_type","surface_form","canonical_label",
        "year","yearly_mean","popularity_weight",
        "context_left","context_right"
    ]

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in joined.iterrows():
            rec = {k: (None if pd.isna(r[k]) else r[k]) for k in fields if k in joined.columns}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
