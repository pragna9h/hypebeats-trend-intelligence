#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pytrends.request import TrendReq
except Exception as e:
    TrendReq = None  # defer hard failure until runtime


# ---------------------------- Logging ---------------------------------
LOGGER = logging.getLogger("clothing_yearly_trends")


# ---------------------------- Helpers ---------------------------------

def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame has a 4-digit integer 'year' column.
    If an index is a DatetimeIndex, use .year; otherwise preserve existing 'year'.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["year"] = df.index.year.astype(int)
    elif "year" in df.columns:
        df["year"] = df["year"].astype(int)
    else:
        raise ValueError("Cannot infer 'year' column; provide datetime index or a 'year' column.")
    return df


@dataclass
class PytrendsConfig:
    locale: str
    tz_minutes: int
    sleep_sec: float


def make_trendreq(cfg: PytrendsConfig) -> TrendReq:
    if TrendReq is None:
        raise RuntimeError("pytrends is not installed. Please `pip install pytrends`.")
    return TrendReq(hl=cfg.locale, tz=cfg.tz_minutes)


# Change signature to accept Sequence[str]
import random
from math import ceil
# utilities
import numpy as np

def _chunk_with_anchor(keywords: list[str], k: int = 5) -> list[list[str]]:
    """
    Return batches of up to k keywords, repeating the first keyword (anchor)
    in every batch for cross-batch normalization.
    """
    kws = list(dict.fromkeys([k.strip() for k in keywords if k.strip()]))  # de-dup, keep order
    if not kws:
        return []
    anchor = kws[0]
    rest = kws[1:]
    batches = []
    if not rest:
        return [[anchor]]
    # each batch = [anchor] + up to (k-1) items
    for i in range(0, len(rest), k - 1):
        batches.append([anchor] + rest[i:i + (k - 1)])
    return batches

def _median_scale(base: pd.Series, other: pd.Series) -> float:
    """Scale `other` to `base` using the median ratio on overlapping, positive points."""
    pair = base.to_frame("b").join(other.rename("o"), how="inner")
    pair = pair[(pair["b"] > 0) & (pair["o"] > 0)]
    if pair.empty:
        return 1.0
    ratios = pair["b"] / pair["o"]
    return float(np.median(ratios))
def _collapse(df: pd.DataFrame, how: str) -> pd.Series:
    if how == "median":
        return df.median(axis=1)
    if how == "mean":
        return df.mean(axis=1)
    # fallback
    return df.max(axis=1)

def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
def fetch_with_retry(py, keywords: list[str], cfg, combine: str, max_tries: int = 5):
    """
    Query keywords in batches of <=5 with an anchor keyword repeated in each batch.
    Rescale each batch to the first batch via the anchor, then collapse using
    the chosen reducer (median/mean/max) to a single composite series.
    """
    batches = _chunk_with_anchor(keywords, k=5)
    if not batches:
        return None

    collected = []
    base_anchor = None

    for batch in batches:
        backoff = 1.0
        for attempt in range(1, max_tries + 1):
            try:
                py.build_payload(batch, timeframe="all")
                df = py.interest_over_time()
                if df is None or df.empty:
                    raise RuntimeError(f"Empty series for {batch}")
                df = df.drop(columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")

                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    raise RuntimeError(f"No numeric series for {batch}")

                anchor_name = batch[0]
                ach = anchor_name if anchor_name in df.columns else numeric_cols[0]
                anchor_series = df[ach].rename("anchor")

                if base_anchor is None:
                    scale = 1.0
                    base_anchor = anchor_series
                else:
                    scale = _median_scale(base_anchor, anchor_series)

                df_scaled = df[numeric_cols] * scale
                # collapse this batch to one series with the chosen reducer
                series = _collapse(df_scaled, combine).to_frame(name="composite")
                collected.append(series)

                # polite pause between batches
                time.sleep(cfg.sleep_sec + random.uniform(0.2, 0.9))
                break

            except Exception as e:
                msg = str(e)
                if "429" in msg:
                    sleep_s = max(10.0, backoff * 5.0) + random.uniform(0.0, 2.0)
                else:
                    sleep_s = backoff + random.uniform(0.0, 0.75)
                time.sleep(sleep_s)
                if attempt >= max_tries:
                    LOGGER.error("Giving up on keywords=%s after %d attempts (%s)", batch, max_tries, e)
                    break
                LOGGER.warning("Attempt %d/%d failed for keywords=%s: %s (sleep %.1fs)",
                               attempt, max_tries, batch, e, sleep_s)
                backoff *= 2

    if not collected:
        return None

    # final collapse across the batch composites
    joined = pd.concat(collected, axis=1)
    comp = _collapse(joined, combine).to_frame(name="composite")
    return comp


def aggregate_period(series: pd.DataFrame, label: str, freq: str) -> pd.DataFrame:
    """Aggregate a single keyword history into yearly or quarterly metrics."""
    if series is None or series.empty:
        return pd.DataFrame()

    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    col = series.columns[0]
    rule = "YS" if freq == "year" else "QS"
    grp = series[col].resample(rule)
    out = pd.DataFrame({
        "trend_mean": grp.mean(),
        "trend_max": grp.max(),
        "trend_min": grp.min(),
        "trend_sum": grp.sum(),
        "n_points": grp.size(),
    })
    if out.empty:
        return pd.DataFrame()

    out["period_start"] = out.index.date
    out["year"] = out.index.year.astype(int)
    if freq == "quarter":
        out["quarter"] = out.index.quarter.astype(int)
        out["period_label"] = out["year"].astype(str) + "Q" + out["quarter"].astype(str)
    else:
        out["period_label"] = out["year"].astype(str)
    out.insert(0, "label", label)

    cols = ["label", "period_start", "period_label", "year"]
    if freq == "quarter":
        cols.append("quarter")
    cols += ["trend_mean", "trend_max", "trend_min", "trend_sum", "n_points"]
    out = out[cols]
    return out


# ---------------------------- Input universe ---------------------------

ALLOWED_TYPES = {"clothing", "material", "accessory"}


def read_labels_from_mentions_parquet(path: str, filter_types: Sequence[str]) -> List[str]:
    df = pd.read_parquet(path)
    if not {"canonical_label", "mention_type"}.issubset(df.columns):
        raise ValueError("mentions parquet must include columns: canonical_label, mention_type")
    flt = df[df["mention_type"].isin(filter_types)]
    labels = (
        flt["canonical_label"].dropna().astype(str).str.strip().str.lower().drop_duplicates().tolist()
    )
    return labels


def read_labels_from_taxonomy(path: str, filter_types: Sequence[str]) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of {canonical_label, type, aliases}
    labels = []
    for row in data:
        t = str(row.get("type", "")).strip().lower()
        if t in filter_types:
            lab = str(row.get("canonical_label", "")).strip().lower()
            if lab:
                labels.append(lab)
    # de-dup while preserving insertion order
    seen = set()
    uniq = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            uniq.append(lab)
    return uniq


# REPLACE read_alias_overrides(...) with this version
def read_alias_overrides(path: Optional[str]) -> Dict[str, list[str]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"label", "query"}.issubset(df.columns):
        raise ValueError("--alias-csv must have columns: label,query")

    out = {}
    for r in df.itertuples(index=False):
        lab = str(r.label).strip().lower()
        q = str(r.query).strip()
        kws = [p.strip().strip('"') for p in q.split(" OR ")] if " OR " in q else [q]
        # de-dup while keeping order
        kws = list(dict.fromkeys(kws))
        out[lab] = kws
    return out


# ---------------------------- Main pipeline ---------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate Google Trends yearly metrics for clothing/material/accessory labels.\n"
            "Mirror of brand_yearly_trends_v2.py with label semantics."
        )
    )

    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--mentions-parquet", dest="mentions_parquet", help="Path to clothing detections parquet")
    gsrc.add_argument("--taxonomy", dest="taxonomy", help="Path to taxonomy JSON with canonical_label/type/aliases")

    p.add_argument("--alias-csv", dest="alias_csv", default=None,
                   help="CSV with columns label,query to override Trends queries")

    p.add_argument("--filter-types", default="clothing,material,accessory",
                   help="Comma-separated subset of {clothing,material,accessory}; default includes all")

    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of labels for quick runs")

    # Common ergonomics preserved from brand script
    p.add_argument("--locale", default="en-US", help="Locale for pytrends hl (default: en-US)")
    p.add_argument("--tz-minutes", type=int, default=0, help="Timezone offset minutes for pytrends tz (default: 0)")
    p.add_argument("--sleep-sec", type=float, default=1.0, help="Sleep seconds between queries (default: 1.0)")
    p.add_argument("--combine", choices=["median", "mean", "max"], default="median", help="How to combine synonym series within a label (default: median).")
    p.add_argument("--freq", choices=["year", "quarter"], default="year",
                   help="Aggregation frequency for the output periods (default: year).")
    p.add_argument("--start-date", default=None,
                   help="Optional ISO date (YYYY-MM-DD); drop trend data before this date.")

    p.add_argument("--checkpoint", action="store_true",
                   help="If set, append/write partial results every ~5 labels to --out")

    p.add_argument("--out", default="data/clothing_yearly_trends.csv",
                   help="Output CSV path (default: data/clothing_yearly_trends.csv)")

    p.add_argument("--no-verbose", action="store_true", help="Reduce log verbosity")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=not args.no_verbose)

    # Parse filter types
    filter_types = [t.strip().lower() for t in str(args.filter_types).split(',') if t.strip()]
    for t in filter_types:
        if t not in ALLOWED_TYPES:
            raise ValueError(f"Invalid filter type '{t}'. Allowed: {sorted(ALLOWED_TYPES)}")

    # Build label universe
    if args.mentions_parquet:
        labels = read_labels_from_mentions_parquet(args.mentions_parquet, filter_types)
    else:
        labels = read_labels_from_taxonomy(args.taxonomy, filter_types)

    if not labels:
        LOGGER.warning("No labels found for filter types %s; exiting without writing output.", filter_types)
        return 0

    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    alias_map = read_alias_overrides(args.alias_csv)

    cfg = PytrendsConfig(locale=args.locale, tz_minutes=args.tz_minutes, sleep_sec=args.sleep_sec)
    py = make_trendreq(cfg)

    all_chunks: List[pd.DataFrame] = []
    written_any = False

    def flush_checkpoint(force: bool = False) -> None:
        nonlocal all_chunks, written_any
        if not all_chunks:
            return
        if not args.checkpoint and not force:
            return
        combined = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
        if combined.empty:
            return
        # Preserve column order as aggregated
        combined = combined.loc[:, combined.columns]
        # Write (overwrite). If you prefer append, de-dup first.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        combined.to_csv(args.out, index=False)
        written_any = True
        LOGGER.info("Checkpoint wrote %d rows to %s", len(combined), args.out)

    batch = 0
    for i, label in enumerate(labels, start=1):
        keywords = alias_map.get(label, [label]) if isinstance(alias_map.get(label), list) else [alias_map.get(label, label)]
        LOGGER.info("[%d/%d] label='%s' | keywords=%s", i, len(labels), label, keywords)
        df = fetch_with_retry(py, keywords, cfg, args.combine)

        if df is None or df.empty:
            LOGGER.warning("Skipping label='%s' due to empty series", label)
        else:
            if args.start_date:
                try:
                    start_dt = pd.to_datetime(args.start_date)
                    df = df[df.index >= start_dt]
                except Exception:
                    LOGGER.warning("Invalid --start-date '%s'; ignoring.", args.start_date)
            ya = aggregate_period(df, label, args.freq)
            if ya.empty:
                LOGGER.warning("No %s rows after resample for label='%s' (skipping)", args.freq, label)
            else:
                all_chunks.append(ya)
                batch += 1
        time.sleep(cfg.sleep_sec)
        if args.checkpoint and batch >= 5:
            flush_checkpoint(force=False)
            batch = 0

    # Final write
    if all_chunks:
        # Combine and write final to ensure completeness
        final_df = pd.concat(all_chunks, ignore_index=True)
        final_df = final_df.loc[:, final_df.columns]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        final_df.to_csv(args.out, index=False)
        LOGGER.info("Wrote final %d rows to %s", len(final_df), args.out)
    else:
        if not written_any:
            LOGGER.warning("No valid labels produced any data; no file written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
