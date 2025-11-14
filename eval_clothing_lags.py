#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, os
from typing import List, Dict
import pandas as pd
import numpy as np

LAGS = [0, 1, 2, 4, 8]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate lag correlations for clothing/material/accessory trends.")
    p.add_argument("--summary-csv", required=True, help="Output from hypebeats_trends_for_clothing.py")
    p.add_argument("--series-dir", required=True, help="Directory passed via --emit-series-dir")
    p.add_argument("--out", default="data/cur/clothing_lag_eval.csv")
    return p.parse_args()

def _safe_name(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["week"] = s["date"] - pd.to_timedelta(s["date"].dt.weekday, unit="D")
    w = s.groupby("week", as_index=False)["value"].mean().rename(columns={"value": "trend"})
    return w.sort_values("week").reset_index(drop=True)

def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 3 or z["x"].std() == 0 or z["y"].std() == 0:
        return np.nan
    return z["x"].corr(z["y"], method="pearson")

def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    """
    Spearman without SciPy: rank both series then Pearson on the ranks.
    """
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 3:
        return np.nan
    xr = z["x"].rank(method="average")
    yr = z["y"].rank(method="average")
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return xr.corr(yr, method="pearson")

def main():
    args = parse_args()
    summ = pd.read_csv(args.summary_csv, parse_dates=["release_date"])
    rows = []

    for _, r in summ.iterrows():
        label = r["label"]
        label_type = r.get("label_type", "")
        song_id = -1 if pd.isna(r["song_id"]) else int(r["song_id"])
        rel = pd.to_datetime(r["release_date"]).date().isoformat()

        # reconstruct series filename (same as trends script)
        key = f"{label}|{label_type}|{song_id}|{rel}"
        fn = f"{_safe_name(key)}.csv"
        path = os.path.join(args.series_dir, fn)
        if not os.path.exists(path):
            rows.append({
                "label": label, "label_type": label_type, "song_id": song_id,
                "title": r.get("title",""), "artist": r.get("artist",""),
                "release_date": rel, "series_file": fn, "note": "series not found",
                **{f"r_pearson_lag_{L}": np.nan for L in LAGS},
                **{f"r_spearman_lag_{L}": np.nan for L in LAGS},
                "best_lag": np.nan, "best_metric": "pearson", "best_r": np.nan
            })
            continue

        daily = pd.read_csv(path, parse_dates=["date"])
        if daily.empty:
            continue
        w = weekly_from_daily(daily)

        # Release-week impulse: 1 at release week, else 0
        release_week = pd.Timestamp(r["release_date"]) - pd.to_timedelta(pd.Timestamp(r["release_date"]).weekday(), unit="D")
        weeks = w["week"]
        x = (weeks == release_week).astype(int)

        pearsons = {}
        spearmans = {}
        for L in LAGS:
            y = w["trend"].shift(-L)  # future trend at lag L
            pearsons[L] = pearson_corr(x, y)
            spearmans[L] = spearman_corr_no_scipy(x, y)

        # choose best by absolute Pearson (change to Spearman if preferred)
        best_lag = max(LAGS, key=lambda L: (-1 if pd.isna(pearsons[L]) else abs(pearsons[L])))
        best_r = pearsons[best_lag]

        rows.append({
            "label": label,
            "label_type": label_type,
            "song_id": song_id,
            "title": r.get("title",""),
            "artist": r.get("artist",""),
            "release_date": rel,
            "series_file": fn,
            **{f"r_pearson_lag_{L}": pearsons[L] for L in LAGS},
            **{f"r_spearman_lag_{L}": spearmans[L] for L in LAGS},
            "best_lag": best_lag,
            "best_metric": "pearson",
            "best_r": best_r,
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows -> {args.out}")
    if len(out):
        print(out[["label","artist","title","best_lag","best_r"]].to_string(index=False))

if __name__ == "__main__":
    main()
