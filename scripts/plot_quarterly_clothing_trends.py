#!/usr/bin/env python3
"""
Build a quick visualization from the quarterly-enriched mentions JSONL.

Example:
  .venv/bin/python plot_quarterly_clothing_trends.py \
      --jsonl data/cur/lyrics_mentions_enriched_v2.jsonl \
      --out plots/top10_quarterly_trends.png \
      --top 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot top clothing labels by average quarterly trend mean")
    p.add_argument("--jsonl", required=True, help="Path to lyrics_mentions_enriched_v2.jsonl")
    p.add_argument("--out", default="plots/top_quarterly_trends.png", help="Output PNG path")
    p.add_argument("--top", type=int, default=10, help="Number of labels to show (default 10)")
    p.add_argument("--min-mentions", type=int, default=5, help="Minimum mentions required per label (default 5)")
    return p.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"{jsonl_path} not found.")

    df = load_jsonl(jsonl_path)
    if "trend_mean" not in df.columns:
        raise SystemExit("JSONL missing 'trend_mean' column. Did you feed merge_mentions_with_trends_v2 output?")

    metric_df = df.dropna(subset=["trend_mean"]).copy()
    if metric_df.empty:
        raise SystemExit("No rows with non-null trend_mean to plot.")

    grouped = (
        metric_df.groupby("canonical_label")
        .agg(
            avg_trend=("trend_mean", "mean"),
            med_trend=("trend_mean", "median"),
            mentions=("canonical_label", "size"),
            popularity_weight=("popularity_weight", "sum"),
        )
        .sort_values("avg_trend", ascending=False)
    )

    grouped = grouped[grouped["mentions"] >= args.min_mentions]
    if grouped.empty:
        raise SystemExit(f"No labels meet min_mentions={args.min_mentions}.")

    top_df = grouped.head(args.top).iloc[::-1]  # reverse for horizontal bar chart

    fig, ax = plt.subplots(figsize=(10, max(4, args.top * 0.4)))
    ax.barh(top_df.index, top_df["avg_trend"], color="#3366cc")
    ax.set_xlabel("Average Quarterly Trend Mean (0-100)")
    ax.set_ylabel("Clothing Label")
    ax.set_title(f"Top {len(top_df)} Clothing Mentions by Quarterly Trend Mean")

    for i, (label, row) in enumerate(top_df.iterrows()):
        ax.text(row["avg_trend"] + 0.5, i, f"{row['avg_trend']:.1f} (n={row['mentions']})", va="center")

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
