#!/usr/bin/env python3
"""
Export Google Trends monthly interest for every taxonomy label and alias with polite pacing.

Example:
  python export_taxonomy_trends.py \
      --taxonomy configs/taxonomy_v2.json \
      --timeframe "2010-01-01 2025-10-28" \
      --geo US \
      --sleep-sec 75 \
      --out data/taxonomy_trends_full.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

try:
    from pytrends.request import TrendReq
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pytrends is required. Install with `pip install pytrends`.") from exc
try:
    from pytrends.exceptions import TooManyRequestsError
except Exception:  # pragma: no cover
    class TooManyRequestsError(Exception):
        """Fallback when pytrends exceptions are unavailable."""
        pass


LOGGER = logging.getLogger("export_taxonomy_trends")
DEFAULT_SLEEP = 75.0
HARD_BLOCK_SLEEP = 900.0
MAX_HARD_ATTEMPTS = 5

_LAST_REQUEST_AT = 0.0


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch month-to-month Google Trends interest for every taxonomy label and alias."
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/taxonomy_v2.json",
        help="Path to taxonomy JSON (default: configs/taxonomy_v2.json).",
    )
    parser.add_argument(
        "--timeframe",
        default="today 5-y",
        help="Google Trends timeframe string (ignored when --start-date is set).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional YYYY-MM-DD start date; splits the range into windows of --window-years.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional YYYY-MM-DD end date (default: today).",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=5,
        help="Span in years per request when using --start-date (default: 5).",
    )
    parser.add_argument(
        "--geo",
        default="",
        help="Geographic code (default: global). e.g., US",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=DEFAULT_SLEEP,
        help="Minimum seconds between requests (default: 75).",
    )
    parser.add_argument(
        "--out",
        default="data/taxonomy_trends.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip keywords already present in --out (based on canonical_label + keyword).",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Optional cap on number of canonical labels (useful for dry runs).",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=None,
        help="Optional cap on total keywords processed (after resume filter).",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Reduce log verbosity.",
    )
    return parser.parse_args()


def load_taxonomy(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Taxonomy JSON must contain a list, got {type(data).__name__}")
    return data


def sanitize_keywords(keywords: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for term in keywords:
        txt = str(term).strip()
        if not txt:
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(txt)
    return ordered


def polite_sleep(min_interval: float, jitter: float = 0.25) -> None:
    global _LAST_REQUEST_AT
    min_interval = max(0.0, float(min_interval))
    now = time.time()
    elapsed = now - _LAST_REQUEST_AT
    wait_for = max(0.0, min_interval - elapsed)
    if wait_for > 0:
        time.sleep(wait_for)
    if jitter > 0:
        extra = random.uniform(0, min_interval * jitter)
        if extra > 0:
            time.sleep(extra)
    _LAST_REQUEST_AT = time.time()


def build_timeframes(
    start_date: str | None,
    end_date: str | None,
    default_timeframe: str,
    window_years: int,
) -> List[str]:
    if not start_date:
        return [default_timeframe]

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize() if end_date else pd.Timestamp.today().normalize()
    if end < start:
        raise ValueError("end-date must be on or after start-date.")
    window_years = max(1, int(window_years))

    frames: List[str] = []
    current_start = start
    while current_start <= end:
        candidate_end = current_start + pd.DateOffset(years=window_years)
        if candidate_end > end:
            candidate_end = end
        if candidate_end <= current_start:
            candidate_end = current_start + pd.DateOffset(months=1)
        frames.append(f"{current_start.date()} {candidate_end.date()}")
        if candidate_end >= end:
            break
        current_start = (candidate_end + pd.Timedelta(days=1)).normalize()
    return frames


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(6))
def fetch_interest(
    pytrends: TrendReq,
    keyword: str,
    timeframe: str,
    geo: str,
    sleep_sec: float,
) -> pd.DataFrame:
    polite_sleep(sleep_sec, jitter=0.20)
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        raise RuntimeError(f"Empty interest data for '{keyword}' timeframe={timeframe}")
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop(columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")
    cleaned.index = pd.to_datetime(cleaned.index)
    return cleaned.resample("MS").mean()


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_completed_pairs(path: str) -> set[Tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["canonical_label", "keyword"])
    except Exception:
        df = pd.read_csv(path)
    pairs = set(zip(df["canonical_label"].astype(str), df["keyword"].astype(str)))
    return pairs


def write_keyword_rows(
    out_path: str,
    canonical_label: str,
    label_type: str,
    keyword: str,
    combined: pd.DataFrame,
) -> int:
    ensure_parent_dir(out_path)
    series = combined[keyword].rename("interest")
    out_df = series.reset_index()
    first_col = out_df.columns[0]
    if first_col != "month":
        out_df = out_df.rename(columns={first_col: "month"})
    out_df = out_df.assign(
        canonical_label=canonical_label,
        keyword=keyword,
        label_type=label_type,
    )
    out_df = out_df[
        ["canonical_label", "keyword", "label_type", "month", "interest"]
    ]
    out_df["month"] = pd.to_datetime(out_df["month"]).dt.date.astype(str)

    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    mode = "a" if file_exists else "w"
    header = not file_exists
    out_df.to_csv(out_path, mode=mode, header=header, index=False)
    return len(out_df)


def iter_taxonomy_keywords(
    taxonomy: List[dict],
    max_labels: int | None,
) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    labels_seen = 0
    for entry in taxonomy:
        canonical = str(entry.get("canonical_label", "")).strip()
        if not canonical:
            continue
        label_type = str(entry.get("type", "")).strip().lower()
        aliases = entry.get("aliases", []) or []
        keywords = sanitize_keywords([canonical, *aliases])
        for kw in keywords:
            rows.append((canonical, kw, label_type))
        labels_seen += 1
        if max_labels is not None and labels_seen >= max_labels:
            break
    return rows


def main() -> int:
    args = parse_args()
    setup_logging(verbose=not args.no_verbose)

    taxonomy = load_taxonomy(args.taxonomy)
    keyword_rows = iter_taxonomy_keywords(taxonomy, args.max_labels)

    if not keyword_rows:
        LOGGER.warning("No keywords found in taxonomy.")
        return 0

    timeframes = build_timeframes(args.start_date, args.end_date, args.timeframe, args.window_years)
    LOGGER.info("Timeframes to fetch: %s", timeframes)

    completed_pairs: set[Tuple[str, str]] = set()
    if args.resume:
        completed_pairs = load_completed_pairs(args.out)
        if completed_pairs:
            LOGGER.info("Resume enabled: skipping %d existing keyword pairs.", len(completed_pairs))

    if args.max_keywords is not None:
        keyword_rows = keyword_rows[: args.max_keywords]

    pytrends = TrendReq(hl="en-US", tz=360, retries=0, backoff_factor=0)

    total = len(keyword_rows)
    processed = 0
    skipped = 0

    for idx, (canonical_label, keyword, label_type) in enumerate(keyword_rows, start=1):
        pair = (canonical_label, keyword)
        if pair in completed_pairs:
            skipped += 1
            LOGGER.info("[%d/%d] Skipping existing keyword '%s' (canonical='%s')", idx, total, keyword, canonical_label)
            continue

        LOGGER.info("[%d/%d] Fetching keyword='%s' (canonical='%s', type='%s')", idx, total, keyword, canonical_label, label_type)

        combined = pd.DataFrame()
        attempt = 0
        while True:
            attempt += 1
            try:
                combined = pd.DataFrame()
                for tf_idx, timeframe in enumerate(timeframes, start=1):
                    LOGGER.info("  timeframe %d/%d: %s", tf_idx, len(timeframes), timeframe)
                    df = fetch_interest(pytrends, keyword, timeframe, args.geo, args.sleep_sec)
                    monthly = aggregate_monthly(df)
                    combined = pd.concat([combined, monthly])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                break
            except RetryError as exc:
                last_exc = exc.last_attempt.exception()
                if isinstance(last_exc, TooManyRequestsError):
                    if attempt >= MAX_HARD_ATTEMPTS:
                        LOGGER.error(
                            "Abandoning keyword '%s' after %d TooManyRequests errors.",
                            keyword,
                            attempt,
                        )
                        combined = pd.DataFrame()
                        break
                    LOGGER.warning(
                        "429 TooManyRequests for '%s'. Hard cool-off %.0fs and rotating session (attempt %d/%d).",
                        keyword,
                        HARD_BLOCK_SLEEP,
                        attempt,
                        MAX_HARD_ATTEMPTS,
                    )
                    polite_sleep(HARD_BLOCK_SLEEP, jitter=0.10)
                    pytrends = TrendReq(hl="en-US", tz=360, retries=0, backoff_factor=0)
                    continue
                LOGGER.error(
                    "Failed keyword '%s' (canonical='%s'): %s",
                    keyword,
                    canonical_label,
                    last_exc,
                )
                combined = pd.DataFrame()
                break
            except TooManyRequestsError:
                if attempt >= MAX_HARD_ATTEMPTS:
                    LOGGER.error(
                        "Abandoning keyword '%s' after %d TooManyRequests errors.",
                        keyword,
                        attempt,
                    )
                    combined = pd.DataFrame()
                    break
                LOGGER.warning(
                    "429 TooManyRequests for '%s'. Hard cool-off %.0fs and rotating session (attempt %d/%d).",
                    keyword,
                    HARD_BLOCK_SLEEP,
                    attempt,
                    MAX_HARD_ATTEMPTS,
                )
                polite_sleep(HARD_BLOCK_SLEEP, jitter=0.10)
                pytrends = TrendReq(hl="en-US", tz=360, retries=0, backoff_factor=0)
                continue
            except Exception as exc:
                LOGGER.error("Failed keyword '%s' (canonical='%s'): %s", keyword, canonical_label, exc)
                combined = pd.DataFrame()
                break

        if combined.empty or keyword not in combined.columns:
            LOGGER.warning("No data to write for keyword '%s'. Skipping.", keyword)
            continue

        try:
            rows_written = write_keyword_rows(args.out, canonical_label, label_type, keyword, combined)
        except Exception as exc:
            LOGGER.error("Failed writing rows for keyword '%s': %s", keyword, exc)
            continue

        processed += 1
        completed_pairs.add(pair)
        LOGGER.info("  wrote %d monthly rows", rows_written)

    LOGGER.info("Done. processed=%d skipped=%d total=%d output=%s", processed, skipped, total, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
