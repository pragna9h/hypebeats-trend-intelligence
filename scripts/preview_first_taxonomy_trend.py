#!/usr/bin/env python3
"""
Preview month-to-month Google Trends interest for the first taxonomy entry and its aliases.

Usage:
  python preview_first_taxonomy_trend.py \
      --taxonomy configs/taxonomy_v2.json \
      --start-date 2010-01-01 \
      --geo "" \
      --sleep-sec 75
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from pytrends.request import TrendReq
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pytrends is required. Install with `pip install pytrends`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch monthly Google Trends interest for the first taxonomy entry + aliases."
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/taxonomy_v2.json",
        help="Path to taxonomy JSON (default: configs/taxonomy_v2.json).",
    )
    parser.add_argument(
        "--keywords",
        default=None,
        help="Comma-separated keywords to fetch instead of reading from taxonomy.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index into taxonomy entries (default: 0 for the first entry).",
    )
    parser.add_argument(
        "--timeframe",
        default="today 5-y",
        help="Google Trends timeframe string (default: today 5-y).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional YYYY-MM-DD start date; overrides --timeframe when provided.",
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
        help="Span (years) per request window when using --start-date/--end-date (default: 5).",
    )
    parser.add_argument(
        "--geo",
        default="",
        help="Geographic code for Trends (default: global).",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=75.0,
        help="Seconds to sleep between successive payloads (default: 75).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of aliases to include. 0 keeps only the canonical label.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Maximum keywords per Trends request (default: 5).",
    )
    return parser.parse_args()


def load_taxonomy(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of entries in taxonomy, got {type(data).__name__}")
    if not data:
        raise ValueError("Taxonomy file is empty.")
    return data


def chunk(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def sanitize_keywords(keywords: Iterable[str], limit: int | None) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for keyword in keywords:
        text = str(keyword).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def build_timeframes(
    start_date: str | None,
    end_date: str | None,
    default_timeframe: str,
    window_years: int,
) -> List[str]:
    if not start_date:
        return [default_timeframe]

    start = pd.to_datetime(start_date).normalize()
    if end_date:
        end = pd.to_datetime(end_date).normalize()
    else:
        end = pd.Timestamp.today().normalize()
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    window_years = max(1, int(window_years))

    frames: List[str] = []
    current_start = start
    while current_start <= end:
        next_end = current_start + pd.DateOffset(years=window_years)
        if next_end > end:
            next_end = end
        if next_end <= current_start:
            next_end = current_start + pd.DateOffset(months=1)
        frame = f"{current_start.date()} {next_end.date()}"
        frames.append(frame)
        if next_end >= end:
            break
        current_start = (next_end + pd.Timedelta(days=1)).normalize()
    return frames


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(6))
def fetch_interest(pytrends: TrendReq, keywords: Sequence[str], timeframe: str, geo: str) -> pd.DataFrame:
    pytrends.build_payload(list(keywords), timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        raise RuntimeError("Empty interest data returned by pytrends.")
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop(columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")
    cleaned.index = pd.to_datetime(cleaned.index)
    monthly = cleaned.resample("MS").mean()
    monthly.index.name = "month"
    return monthly


def main() -> int:
    args = parse_args()

    manual_keywords: List[str] | None = None
    if args.keywords:
        manual_keywords = sanitize_keywords([k for k in args.keywords.split(",")], None)
        if not manual_keywords:
            raise ValueError("No usable keywords parsed from --keywords.")
        print(f"Using manual keywords: {manual_keywords}")

    if manual_keywords is None:
        taxonomy_path = Path(args.taxonomy)
        entries = load_taxonomy(str(taxonomy_path))
        if args.index < 0 or args.index >= len(entries):
            raise IndexError(f"--index {args.index} out of range for taxonomy with {len(entries)} entries.")

        entry = entries[args.index]
        canonical = str(entry.get("canonical_label", "")).strip()
        aliases = entry.get("aliases", []) or []

        if not canonical:
            raise ValueError("Selected taxonomy entry is missing a canonical_label.")

        if args.limit is not None and args.limit < 0:
            raise ValueError("--limit must be >= 0")

        limit = None
        if args.limit is not None:
            limit = args.limit + 1  # include canonical label

        keywords = sanitize_keywords([canonical, *aliases], limit)
        if not keywords:
            print("No keywords available after sanitization.", file=sys.stderr)
            return 1

        print(f"Selected entry: {canonical!r}")
        if aliases:
            print(f"Aliases ({len(aliases)} total): {aliases[:10]}{'...' if len(aliases) > 10 else ''}")
        print(f"Fetching keywords (deduped, limit applied): {keywords}")
    else:
        keywords = manual_keywords
        print(f"Fetching manual keyword set: {keywords}")
    timeframes = build_timeframes(args.start_date, args.end_date, args.timeframe, args.window_years)
    print(f"Timeframes={timeframes}, geo={args.geo or 'Global'}, chunk_size={args.chunk_size}")

    request_plan: List[Tuple[str, List[str]]] = []
    max_chunk = max(1, args.chunk_size)
    for tf in timeframes:
        for group in chunk(keywords, max_chunk):
            request_plan.append((tf, group))

    if not request_plan:
        print("No requests to execute (empty keyword/timeframe plan).", file=sys.stderr)
        return 1

    pytrends = TrendReq(hl="en-US", tz=360, retries=0, backoff_factor=0)

    combined = pd.DataFrame()
    total_requests = len(request_plan)
    for idx, (tf, group) in enumerate(request_plan, start=1):
        print(f"\nRequest {idx}/{total_requests}: timeframe={tf} | keywords={group}")
        df = fetch_interest(pytrends, group, tf, args.geo)
        monthly = aggregate_monthly(df)
        print(monthly.to_string())

        combined = pd.concat([combined, monthly])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        if idx < total_requests:
            sleep_sec = max(0.0, float(args.sleep_sec))
            print(f"Sleeping {sleep_sec:.1f}s to respect pacing...")
            time.sleep(sleep_sec)

    print("\nCombined monthly interest:")
    print(combined.to_string())
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
