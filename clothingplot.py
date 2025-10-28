#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TITLE = "Top 10 Fastest-Growing Clothing Items by Google Trends Popularity"
YLABEL = "Yearly Mean Interest (0–100)"
LEGEND_TITLE = "Clothing Item"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Top-10 growth lines for clothing labels (yearly means)")
    p.add_argument("--csv", default="data/clothing_yearly_trends.csv",
                   help="Input CSV (default: data/clothing_yearly_trends.csv)")
    p.add_argument("--out", default=None, help="Optional output image path (e.g., clothing_top10_growth.png)")
    p.add_argument("--window-years", type=int, default=0,
                   help="If >0, growth = mean(last N years) - mean(first N years) using overlapping window.")
    p.add_argument("--metric", choices=["end-start", "slope"], default="end-start",
                   help="Growth metric: end-start (default) or slope via numpy.polyfit.")
    return p.parse_args(argv)


def label_growth(df_label: pd.DataFrame, window_years: int = 0, metric: str = "end-start") -> float:
    """Compute growth for one label using the chosen metric/window."""
    df_label = df_label.sort_values("year")
    years = df_label["year"].to_numpy()
    vals = df_label["yearly_mean"].to_numpy()

    # Windowed means if requested and we have enough points
    if window_years and len(vals) >= 2 * window_years:
        start_mean = float(np.mean(vals[:window_years]))
        end_mean = float(np.mean(vals[-window_years:]))
        return end_mean - start_mean

    # Robust-ish trend via slope of best-fit line
    if metric == "slope" and len(vals) >= 3:
        slope, _intercept = np.polyfit(years, vals, 1)
        return float(slope)

    # Default: full-span end - start
    if len(vals) >= 2:
        return float(vals[-1] - vals[0])

    # Not enough data to compute growth
    return float("-inf")


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        logging.error("Input CSV not found: %s", args.csv)
        return 2

    required = {"label", "year", "yearly_mean"}
    if not required.issubset(df.columns):
        logging.error("CSV must contain columns: %s", sorted(required))
        return 2

    # Clean types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["label"] = df["label"].astype(str)
    df["yearly_mean"] = pd.to_numeric(df["yearly_mean"], errors="coerce")
    df = df.dropna(subset=["year", "yearly_mean"]).copy()
    df["year"] = df["year"].astype(int)

    if df.empty:
        logging.warning("No rows to plot.")
        return 0

    # Compute growth per label using chosen metric/window
    growth_rows = []
    for label, g in df.groupby("label"):
        if len(g) < 2:
            continue
        grow = label_growth(g, window_years=args.window_years, metric=args.metric)
        if np.isfinite(grow):
            growth_rows.append((label, grow))

    if not growth_rows:
        logging.warning("No labels with computable growth.")
        return 0

    growth_rows.sort(key=lambda x: x[1], reverse=True)
    top_labels = [l for l, _ in growth_rows[:10]]

    # Subset to top labels and pivot for plotting
    sub = df[df["label"].isin(top_labels)].copy()
    pivot = sub.pivot_table(index="year", columns="label", values="yearly_mean", aggfunc="mean").sort_index()

    if pivot.empty:
        logging.warning("Nothing to plot after pivot.")
        return 0

    # Plot
    plt.figure(figsize=(10, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=col)

    subtitle = ""
    if args.window_years:
        subtitle += f" (window={args.window_years}y)"
    if args.metric != "end-start":
        subtitle += f" [{args.metric}]"

    plt.title(TITLE + subtitle)
    plt.xlabel("Year")
    plt.ylabel(YLABEL)
    plt.legend(title=LEGEND_TITLE, loc="best")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        logging.info("Saved figure to %s", args.out)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
