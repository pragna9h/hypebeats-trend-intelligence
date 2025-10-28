#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Join mention data with quarterly trends to produce enriched JSONL")
    p.add_argument("--mentions", required=True, help="Input mentions parquet (taxonomy v2)")
    p.add_argument("--trends", required=True, help="Quarterly trends CSV from clothing_yearly_trends.py (--freq quarter)")
    p.add_argument("--min-pageviews", type=int, default=150_000, help="Filter mentions below this Genius pageview count")
    p.add_argument("--only-hot", action="store_true", help="Keep only mentions from songs marked as hot on Genius")
    p.add_argument("--out", default="data/cur/lyrics_mentions_enriched_v2.jsonl", help="Output JSONL path")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    m = pd.read_parquet(args.mentions)
    t = pd.read_csv(args.trends)

    # Basic normalization
    m["canonical_label"] = m["canonical_label"].astype(str).str.lower().str.strip()
    m["release_date"] = pd.to_datetime(m["release_date"], errors="coerce")
    m["pageviews"] = pd.to_numeric(m.get("pageviews", np.nan), errors="coerce")
    m["hot"] = m.get("hot", False).astype(bool)

    # Filters
    if args.min_pageviews:
        before = len(m)
        m = m[m["pageviews"].fillna(0) >= args.min_pageviews]
        print(f"Pageviews filter ({args.min_pageviews}+): kept {len(m)}, dropped {before - len(m)}")

    if args.only_hot:
        before = len(m)
        m = m[m["hot"]]
        print(f"Hot filter: kept {len(m)}, dropped {before - len(m)}")

    if m.empty:
        print("No mentions after filtering; nothing to write.")
        return 0

    # Prepare join keys (year+quarter)
    m = m[m["release_date"].notna()].copy()
    if m.empty:
        print("No mentions with valid release dates after filtering; nothing to write.")
        return 0
    m["year"] = m["release_date"].dt.year
    m["quarter"] = m["release_date"].dt.quarter
    m["period_label"] = m["year"].astype(str) + "Q" + m["quarter"].astype(str)

    t["label"] = t["label"].astype(str).str.lower().str.strip()
    t["period_label"] = t["period_label"].astype(str).str.strip()

    join_cols = ["label", "period_label"]
    t_cols = join_cols + [c for c in t.columns if c not in join_cols]

    joined = m.merge(
        t[t_cols],
        left_on=["canonical_label", "period_label"],
        right_on=["label", "period_label"],
        how="left",
    )

    if joined.empty:
        print("No mentions could be matched to quarterly trends; nothing to write.")
        return 0

    joined["popularity_weight"] = np.log1p(joined["pageviews"].fillna(0)).astype(float)

    fields = [
        "song_id", "artist", "title", "release_date", "pageviews", "hot",
        "mention_type", "surface_form", "canonical_label",
        "period_label", "trend_mean", "trend_max", "trend_min", "trend_sum",
        "popularity_weight", "context_window"
    ]

    n = 0
    out_path = args.out
    def _json_ready(val):
        if isinstance(val, pd.Timestamp):
            if pd.isna(val):
                return None
            return val.date().isoformat()
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            v = float(val)
            if np.isnan(v):
                return None
            return v
        return val

    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in joined.iterrows():
            rec = {}
            for k in fields:
                if k not in joined.columns:
                    continue
                val = r.get(k)
                rec[k] = _json_ready(val)
            # add release quarter metadata useful downstream
            rec.setdefault("release_quarter", rec.get("period_label"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
