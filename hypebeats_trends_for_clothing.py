#!/usr/bin/env python3
"""
Fetch Google Trends around release dates for clothing/material/accessory labels
seen in your mentions parquet.

Requires:
  pip install pytrends pandas pyarrow python-dateutil
"""

from __future__ import annotations
import argparse
import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

import pandas as pd
from dateutil import parser as dateparser
from pytrends.request import TrendReq
import random


# ----------------------------
# Args
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Google Trends around song release dates for clothing/material/accessory mentions."
    )
    p.add_argument("--parquet", required=True, help="Path to mentions parquet.")
    p.add_argument("--out", required=True, help="Output CSV path for window stats.")
    p.add_argument("--window-days", type=int, default=14, help="Half-window in days before/after release date.")
    p.add_argument("--sleep-sec", type=float, default=1.0, help="Seconds to sleep between pytrends calls.")
    p.add_argument("--max-tries", type=int, default=5, help="Maximum attempts per query window before giving up.")
    p.add_argument("--label-alias-csv", default=None, help="Optional CSV with columns [label,query].")
    p.add_argument("--timezone-minutes", type=int, default=0, help="Timezone offset minutes for pytrends (e.g., 420).")
    p.add_argument("--language", default="en-US", help="Locale for pytrends.")
    p.add_argument("--min-samples", type=int, default=5, help="Min daily points required before/after to compute stats.")
    p.add_argument("--filter-types", default="clothing,material,accessory", help="Types to include, comma-separated.")
    p.add_argument("--label-col", default="canonical_label", help="Column containing the query label.")
    p.add_argument(
        "--emit-series-dir",
        default=None,
        help="If set, save per-(label,song) daily trend series CSVs into this directory.",
    )
    return p.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def _safe_name(s: str) -> str:
    """File-safe short name for series files."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"label", "query"} <= set(df.columns):
        raise SystemExit("Alias CSV must have columns: label, query")
    return dict(zip(df["label"].astype(str), df["query"].astype(str)))


def safe_parse_date(v) -> Optional[pd.Timestamp]:
    if pd.isna(v):
        return None
    try:
        dt = dateparser.parse(str(v), default=datetime(1970, 1, 1))
        return pd.to_datetime(dt.date())
    except Exception:
        return None


def compute_window_stats(series: pd.Series, split_date: pd.Timestamp) -> Dict[str, float]:
    """
    series: daily interest scores (0..100), DatetimeIndex
    split_date: release date; 'before' is < split_date; 'after' is >= split_date
    """
    before = series[series.index < split_date]
    after = series[series.index >= split_date]

    res = {
        "trend_mean_before": float("nan"),
        "trend_mean_after": float("nan"),
        "trend_delta": float("nan"),
        "trend_ratio": float("nan"),
        "trend_peak_after": float("nan"),
        "peak_after_date": pd.NaT,
        "peak_after_lag_days": float("nan"),
        "n_before_points": int(before.shape[0]),
        "n_after_points": int(after.shape[0]),
    }
    if before.shape[0] > 0:
        res["trend_mean_before"] = float(before.mean())
    if after.shape[0] > 0:
        res["trend_mean_after"] = float(after.mean())
        res["trend_peak_after"] = float(after.max())
        if not after.empty:
            peak_idx = after.idxmax()
            res["peak_after_date"] = peak_idx
            res["peak_after_lag_days"] = float((peak_idx - split_date).days)

    if pd.notna(res["trend_mean_before"]) and pd.notna(res["trend_mean_after"]):
        res["trend_delta"] = res["trend_mean_after"] - res["trend_mean_before"]
        res["trend_ratio"] = (
            res["trend_mean_after"] / res["trend_mean_before"] if res["trend_mean_before"] > 0 else float("inf")
        )
    return res


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    df = pd.read_parquet(args.parquet)
    required = {"release_date", "song_id", "title", "artist", args.label_col, "mention_type"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Parquet is missing required columns: {missing}")

    df = df.copy()
    df["release_date_parsed"] = df["release_date"].apply(safe_parse_date)
    keep_types: Set[str] = {t.strip().lower() for t in args.filter_types.split(",") if t.strip()}
    df = df[
        df["release_date_parsed"].notna()
        & df["mention_type"].astype(str).str.lower().isin(keep_types)
        & df[args.label_col].notna()
        & (df[args.label_col].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    if df.empty:
        print("No rows with valid label, mention_type, and release_date. Nothing to do.")
        return

    keys = df[[args.label_col, "mention_type", "release_date_parsed", "song_id", "title", "artist"]].drop_duplicates()
    print(f"Unique (label, type, release_date, song_id) pairs: {len(keys)}")

    alias_map = load_alias_map(args.label_alias_csv)
    pytrends = TrendReq(hl=args.language, tz=args.timezone_minutes)

    out_rows = []
    half = timedelta(days=args.window_days)

    # Ensure series dir exists if requested
    if args.emit_series_dir:
        os.makedirs(args.emit_series_dir, exist_ok=True)

    for _, row in keys.iterrows():
        label = str(row[args.label_col]).strip()
        label_type = str(row["mention_type"]).strip().lower()
        term = alias_map.get(label, label)
        release_date = row["release_date_parsed"]
        song_id = row["song_id"]
        title = row["title"]
        artist = row["artist"]
        if not term:
            continue

        start = (release_date - half).strftime("%Y-%m-%d")
        end = (release_date + half).strftime("%Y-%m-%d")

        success = False
        backoff = max(args.sleep_sec, 1.0)

        for attempt in range(1, args.max_tries + 1):
            try:
                pytrends.build_payload([term], timeframe=f"{start} {end}")
                trend_df = pytrends.interest_over_time()

                if trend_df.empty or term not in trend_df.columns:
                    print(f"[WARN] No trend data for '{term}' around {release_date.date()}")
                    success = True  # nothing to do but don't retry
                    break

                series = trend_df[term]
                series.index = pd.to_datetime(series.index.date)

                # optionally dump the raw daily series
                if args.emit_series_dir:
                    key = f"{label}|{label_type}|{int(song_id) if pd.notna(song_id) else -1}|{release_date.date().isoformat()}"
                    fn = f"{_safe_name(key)}.csv"
                    series.to_frame("value").to_csv(os.path.join(args.emit_series_dir, fn), index_label="date")

                stats = compute_window_stats(series, release_date)
                if stats["n_before_points"] < args.min_samples or stats["n_after_points"] < args.min_samples:
                    print(
                        f"[SKIP] Not enough data for '{term}' around {release_date.date()} "
                        f"(before={stats['n_before_points']}, after={stats['n_after_points']})"
                    )
                    success = True
                    break

                out_rows.append(
                    {
                        "label": label,
                        "label_type": label_type,
                        "query_used": term,
                        "song_id": song_id,
                        "title": title,
                        "artist": artist,
                        "release_date": release_date.date().isoformat(),
                        **stats,
                    }
                )
                success = True
                break

            except Exception as e:
                msg = str(e)
                if attempt >= args.max_tries:
                    print(f"[ERROR] {label} @ {release_date.date()}: {msg}")
                    break
                if "429" in msg:
                    backoff = max(backoff * 2, args.sleep_sec * 2)
                else:
                    backoff = backoff + 0.5
                wait = backoff + random.uniform(0.5, 1.5)
                print(f"[RETRY] Attempt {attempt}/{args.max_tries} for '{term}' hit error: {msg}. Sleeping {wait:.1f}s")
                time.sleep(wait)

        # gentle pause between labels even when successful
        if success:
            time.sleep(args.sleep_sec + random.uniform(0.0, 0.5))

    if not out_rows:
        print("No trend rows produced. Check your labels/queries and dates.")
        return

    out_df = pd.DataFrame(out_rows).sort_values(["label", "release_date", "song_id"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
