#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None  # defer hard failure until runtime


LOGGER = logging.getLogger("clothing_yearly_trends")

DEFAULT_MIN_INTERVAL = 18.0
SESSION_TIMEOUT = (10, 30)

_SESSION = None
_SESSION_SETTINGS: Tuple[str, int] | None = None
_LAST_REQUEST_AT = 0.0


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def canonicalize_label(label: str) -> str:
    return str(label).strip().lower()


def new_session(locale: str = "en-US", tz_minutes: int = 360) -> TrendReq:
    if TrendReq is None:
        raise RuntimeError("pytrends is not installed. Please `pip install pytrends`.")
    return TrendReq(hl=locale, tz=tz_minutes, retries=0, backoff_factor=0, timeout=SESSION_TIMEOUT)


def polite_sleep(base_seconds: float, jitter: float = 0.20) -> None:
    global _LAST_REQUEST_AT
    base_seconds = max(0.0, float(base_seconds))
    now = time.time()
    target = max(now, _LAST_REQUEST_AT + base_seconds)
    sleep_for = target - now
    if sleep_for > 0:
        time.sleep(sleep_for)
    delta = max(0.0, base_seconds * float(jitter))
    if delta > 0:
        time.sleep(random.uniform(0.0, delta))
    _LAST_REQUEST_AT = time.time()


def fetch_interest(
    labels: Sequence[str],
    timeframe: str,
    geo: str = "",
    cat: int = 0,
    gprop: str = "",
    *,
    base_delay: float = DEFAULT_MIN_INTERVAL,
    locale: str = "en-US",
    tz_minutes: int = 360,
    backoff_cap: float = 600.0,
    non429_limit: int = 3,
) -> pd.DataFrame:
    global _SESSION, _SESSION_SETTINGS

    cleaned: List[str] = []
    seen = set()
    for label in labels:
        term = str(label).strip()
        if not term or term in seen:
            continue
        cleaned.append(term)
        seen.add(term)

    if not cleaned:
        raise ValueError("fetch_interest requires at least one non-empty term.")

    base_delay = max(DEFAULT_MIN_INTERVAL, float(base_delay))
    backoff_cap = 600.0  # hard cap at 10 minutes
    non429_limit = max(1, int(non429_limit))

    if _SESSION is None or _SESSION_SETTINGS != (locale, tz_minutes):
        _SESSION = new_session(locale=locale, tz_minutes=tz_minutes)
        _SESSION_SETTINGS = (locale, tz_minutes)
    pytrends = _SESSION

    consecutive_429 = 0
    non429_errors = 0
    backoff = 60.0
    attempt = 0

    while True:
        attempt += 1
        try:
            polite_sleep(base_delay, jitter=0.25)
            pytrends.build_payload(cleaned, timeframe=timeframe, geo=geo, cat=cat, gprop=gprop)
            df = pytrends.interest_over_time()
            if df is None:
                raise RuntimeError("pytrends.interest_over_time returned None")
            consecutive_429 = 0
            return df
        except Exception as exc:
            msg = str(exc)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg)
            if is_429:
                consecutive_429 += 1
                LOGGER.warning("429 on attempt %d for %s; consecutive=%d", attempt, cleaned, consecutive_429)
                if consecutive_429 >= 3:
                    cooloff = 900.0
                    LOGGER.warning("Hard block suspected. Cooling off for %ss and rotating session.", int(cooloff))
                    polite_sleep(cooloff, jitter=0.10)
                    _SESSION = new_session(locale=locale, tz_minutes=tz_minutes)
                    _SESSION_SETTINGS = (locale, tz_minutes)
                    pytrends = _SESSION
                    consecutive_429 = 0
                    backoff = 60.0
                    continue
                LOGGER.warning("Cooling off for %ss before retry.", int(backoff))
                polite_sleep(backoff, jitter=0.15)
                backoff = min(backoff * 2, backoff_cap)
                continue

            non429_errors += 1
            LOGGER.warning("Non-429 error on attempt %d for %s: %s", attempt, cleaned, msg)
            if non429_errors < non429_limit:
                polite_sleep(30.0, jitter=0.25)
                continue
            raise


def _collapse(df: pd.DataFrame, how: str) -> pd.Series:
    if how == "median":
        return df.median(axis=1)
    if how == "mean":
        return df.mean(axis=1)
    return df.max(axis=1)


def combine_synonym_series(df: pd.DataFrame, combine: str) -> pd.Series:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric series returned by pytrends.")
    return _collapse(df[numeric_cols], combine)


def aggregate_period(series: pd.DataFrame, label: str, freq: str) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame()

    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    col = series.columns[0]
    rule = "YS" if freq == "year" else "QS"
    grp = series[col].resample(rule)
    out = pd.DataFrame(
        {
            "trend_mean": grp.mean(),
            "trend_max": grp.max(),
            "trend_min": grp.min(),
            "trend_sum": grp.sum(),
            "n_points": grp.size(),
        }
    )
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

    columns = ["label", "period_start", "period_label", "year"]
    if freq == "quarter":
        columns.append("quarter")
    columns += ["trend_mean", "trend_max", "trend_min", "trend_sum", "n_points"]
    return out[columns]


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
    labels: List[str] = []
    for row in data:
        t = str(row.get("type", "")).strip().lower()
        if t in filter_types:
            lab = str(row.get("canonical_label", "")).strip().lower()
            if lab:
                labels.append(lab)
    seen = set()
    uniq = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            uniq.append(lab)
    return uniq


def read_alias_overrides(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"label", "query"}.issubset(df.columns):
        raise ValueError("--alias-csv must have columns: label,query")

    overrides: Dict[str, List[str]] = {}
    for row in df.itertuples(index=False):
        label = canonicalize_label(row.label)
        query = str(row.query).strip()
        if not query:
            continue
        parts = [p.strip().strip('"') for p in query.split(" OR ")] if " OR " in query else [query]
        cleaned: List[str] = []
        seen_terms = set()
        for part in parts:
            if not part:
                continue
            if part in seen_terms:
                continue
            cleaned.append(part)
            seen_terms.add(part)
        if not cleaned:
            continue
        existing = overrides.get(label, [])
        for term in cleaned:
            if term not in existing:
                existing.append(term)
        overrides[label] = existing[:5]
    return overrides


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def append_dataframe_to_csv(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if file_exists else "w"
    df.to_csv(path, mode=mode, header=not file_exists, index=False)


def append_checkpoint_record(path: str, row: Dict[str, str], fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed_keys(
    checkpoint_path: str,
    output_path: str,
    timeframe: str,
    geo: str,
) -> set[Tuple[str, str, str]]:
    completed: set[Tuple[str, str, str]] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("label", "")
                tf = row.get("timeframe", timeframe)
                g = row.get("geo", geo)
                if label:
                    completed.add((canonicalize_label(label), tf, g or ""))
    if output_path and os.path.exists(output_path):
        try:
            labels = pd.read_csv(output_path, usecols=["label"])["label"]
        except Exception:
            labels = pd.read_csv(output_path)["label"]
        for label in labels.dropna().astype(str):
            completed.add((canonicalize_label(label), timeframe, geo or ""))
    return completed


def build_label_queries(
    labels: Sequence[str],
    alias_map: Dict[str, List[str]],
    max_terms: int = 5,
) -> "OrderedDict[str, List[str]]":
    mapping: "OrderedDict[str, List[str]]" = OrderedDict()
    for label in labels:
        canonical = canonicalize_label(label)
        if canonical in mapping:
            continue
        queries = list(alias_map.get(canonical, []))
        base_term = str(label).strip()
        ordered_terms: List[str] = []
        for term in [base_term, *queries]:
            term = str(term).strip()
            if not term or term in ordered_terms:
                continue
            ordered_terms.append(term)
        if not ordered_terms:
            ordered_terms.append(canonical)
        mapping[canonical] = ordered_terms[:max_terms]
    return mapping


def prepare_aggregated_frame(
    df: pd.DataFrame,
    label: str,
    queries: Sequence[str],
    combine: str,
    freq: str,
    start_dt: Optional[pd.Timestamp],
) -> pd.DataFrame:
    working = df.drop(columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")
    combined = combine_synonym_series(working, combine)
    series = combined.to_frame(name="composite")
    if start_dt is not None:
        series = series[series.index >= start_dt]
    aggregated = aggregate_period(series, label, freq)
    if aggregated.empty:
        raise ValueError("no aggregated rows")
    aggregated = aggregated.copy()
    aggregated["query_count"] = len(queries)
    return aggregated


def write_outputs(
    aggregated: pd.DataFrame,
    args: argparse.Namespace,
    timeframe: str,
    geo: str,
    source_batch: str,
    checkpoint_fields: Sequence[str],
) -> int:
    frame = aggregated.copy()
    frame.insert(1, "timeframe", timeframe)
    frame.insert(2, "geo", geo or "")
    rows = len(frame)
    append_dataframe_to_csv(frame, args.out)

    label_value = frame["label"].iloc[0] if rows else ""
    checkpoint_row = {
        "label": label_value,
        "timeframe": timeframe,
        "geo": geo or "",
        "done_at": datetime.now(timezone.utc).isoformat(),
        "source_batch": source_batch,
    }
    try:
        append_checkpoint_record(args.checkpoint_path, checkpoint_row, checkpoint_fields)
    except Exception as exc:
        LOGGER.warning("Unable to update checkpoint for label '%s': %s", label_value, exc)
    return rows


def attempt_label(
    label: str,
    queries: Sequence[str],
    timeframe: str,
    geo: str,
    args: argparse.Namespace,
    base_delay: float,
    backoff_cap: float,
    max_non429: int,
    start_dt: Optional[pd.Timestamp],
    source_batch: str,
    checkpoint_fields: Sequence[str],
    local_cache: Dict[Tuple[str, str, str], pd.DataFrame],
    use_cache: bool = True,
) -> Tuple[bool, Optional[str], int]:
    cache_key = (label, timeframe, geo or "")
    df: Optional[pd.DataFrame] = None
    if use_cache:
        cached = local_cache.get(cache_key)
        if cached is not None:
            df = cached.copy()

    if df is None:
        try:
            df = fetch_interest(
                queries,
                timeframe,
                geo=geo,
                base_delay=base_delay,
                locale=args.locale,
                tz_minutes=args.tz_minutes,
                backoff_cap=backoff_cap,
                non429_limit=max_non429,
            )
        except Exception as exc:
            return False, str(exc), 0
        local_cache[cache_key] = df.copy()

    try:
        aggregated = prepare_aggregated_frame(df, label, queries, args.combine, args.freq, start_dt)
    except Exception as exc:
        return False, f"aggregation failed: {exc}", 0

    try:
        rows = write_outputs(aggregated, args, timeframe, geo, source_batch, checkpoint_fields)
    except Exception as exc:
        return False, f"write failed: {exc}", 0

    return True, None, rows


def emit_coverage_report(
    total: int,
    pre_completed: int,
    new_successes: int,
    failures: Dict[str, Optional[str]],
    coverage_to_stdout: bool,
) -> None:
    done = pre_completed + new_successes
    pct = (done / total * 100.0) if total else 100.0
    LOGGER.info(
        "Coverage %d/%d labels complete (%.1f%%). pre-completed=%d, newly fetched=%d",
        done,
        total,
        pct,
        pre_completed,
        new_successes,
    )
    if failures:
        LOGGER.error("Failed labels (%d):", len(failures))
        for label, error in sorted(failures.items()):
            LOGGER.error("  %s -> %s", label, error)
    else:
        LOGGER.info("All labels fetched successfully.")

    if coverage_to_stdout:
        print(f"Coverage {done}/{total} labels complete ({pct:.1f}%).")
        if failures:
            print("Failed labels:")
            for label, error in sorted(failures.items()):
                print(f"  {label}: {error}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Google Trends yearly metrics for clothing/material/accessory labels "
            "with restart-safe checkpointing."
        )
    )

    gsrc = parser.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--mentions-parquet", dest="mentions_parquet", help="Path to clothing detections parquet")
    gsrc.add_argument("--taxonomy", dest="taxonomy", help="Path to taxonomy JSON with canonical_label/type/aliases")

    parser.add_argument("--alias-csv", dest="alias_csv", default=None, help="CSV with columns label,query")
    parser.add_argument(
        "--filter-types",
        default="clothing,material,accessory",
        help="Comma-separated subset of {clothing,material,accessory}; default includes all",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of labels for quick runs")

    parser.add_argument("--locale", default="en-US", help="Locale for pytrends hl (default: en-US)")
    parser.add_argument("--tz-minutes", type=int, default=360, help="Timezone offset minutes for pytrends tz (default: 360)")
    parser.add_argument("--sleep-sec", type=float, default=18.0, help="Minimum inter-request seconds (default: 18.0)")
    parser.add_argument("--timeframe", default="all", help="Google Trends timeframe string (default: all)")
    parser.add_argument("--geo", default="", help="Geographic code for pytrends (default: global)")
    parser.add_argument("--combine", choices=["median", "mean", "max"], default="median", help="How to combine synonym series within a label.")
    parser.add_argument("--freq", choices=["year", "quarter"], default="year", help="Aggregation frequency for the output periods.")
    parser.add_argument("--start-date", default=None, help="Optional ISO date (YYYY-MM-DD); drop trend data before this date.")
    parser.add_argument("--max-tries", type=int, default=3, help="Max non-429 retries before surfacing the error.")
    parser.add_argument("--retry-multiplier", type=float, default=2.0, help="Retained for compatibility; exponential factor is fixed at 2.")
    parser.add_argument("--retry-max-sleep", type=float, default=600.0, help="Retained for compatibility; hard cap is 600 seconds.")
    parser.add_argument("--post429-sleep", type=float, default=0.0, help="Retained for compatibility; handled internally.")
    parser.add_argument("--checkpoint", action="store_true", help="Retained for compatibility; checkpointing is always enabled.")

    parser.add_argument("--checkpoint-path", default="checkpoints/fetched_labels.csv", help="Checkpoint file path (default: checkpoints/fetched_labels.csv)")
    parser.add_argument("--out", default="data/clothing_yearly_trends.csv", help="Output CSV path (default: data/clothing_yearly_trends.csv)")
    parser.add_argument("--coverage-report", action="store_true", help="Print coverage summary to stdout in addition to logs.")
    parser.add_argument("--no-verbose", action="store_true", help="Reduce log verbosity.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=not args.no_verbose)

    if TrendReq is None:
        raise RuntimeError("pytrends is not installed. Please `pip install pytrends`.")

    if args.retry_multiplier != 2.0:
        LOGGER.info("Ignoring --retry-multiplier override; exponential backoff factor fixed at 2.")
    if args.retry_max_sleep != 600.0:
        LOGGER.info("Overriding --retry-max-sleep to 600s cap to satisfy rate limit policy.")
    if args.post429_sleep:
        LOGGER.info("--post429-sleep handled by hard-block logic; explicit value ignored.")
    if args.checkpoint:
        LOGGER.info("Checkpointing is always enabled; --checkpoint flag retained for compatibility.")

    filter_types = [t.strip().lower() for t in str(args.filter_types).split(",") if t.strip()]
    for filter_type in filter_types:
        if filter_type not in ALLOWED_TYPES:
            raise ValueError(f"Invalid filter type '{filter_type}'. Allowed: {sorted(ALLOWED_TYPES)}")

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
    label_queries = build_label_queries(labels, alias_map)
    label_list = list(label_queries.keys())

    if not label_list:
        LOGGER.warning("No labels remain after alias consolidation.")
        return 0

    timeframe = str(args.timeframe)
    geo = str(args.geo or "")
    base_delay = max(DEFAULT_MIN_INTERVAL, float(args.sleep_sec))
    backoff_cap = 600.0
    max_non429 = max(1, int(args.max_tries))

    checkpoint_fields = ["label", "timeframe", "geo", "done_at", "source_batch"]
    source_path = args.mentions_parquet or args.taxonomy
    source_batch = Path(source_path).name if source_path else "unknown"

    completed_keys = load_completed_keys(args.checkpoint_path, args.out, timeframe, geo)
    pre_completed = sum(1 for label in label_list if (label, timeframe, geo) in completed_keys)
    if pre_completed:
        LOGGER.info("Skipping %d labels already present in output/checkpoint.", pre_completed)

    start_dt: Optional[pd.Timestamp] = None
    if args.start_date:
        try:
            start_dt = pd.to_datetime(args.start_date)
        except Exception:
            LOGGER.warning("Invalid --start-date '%s'; ignoring.", args.start_date)
            start_dt = None

    local_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    successes = 0
    failures: Dict[str, Optional[str]] = {}

    total_labels = len(label_list)
    for idx, label in enumerate(label_list, start=1):
        queries = label_queries[label]
        LOGGER.info("[%d/%d] label='%s' | queries=%s", idx, total_labels, label, queries)
        key = (label, timeframe, geo)
        if key in completed_keys:
            LOGGER.info("Skipping label '%s'; already completed.", label)
            continue

        success, error, rows = attempt_label(
            label,
            queries,
            timeframe,
            geo,
            args,
            base_delay,
            backoff_cap,
            max_non429,
            start_dt,
            source_batch,
            checkpoint_fields,
            local_cache,
            use_cache=True,
        )
        if success:
            completed_keys.add(key)
            successes += 1
            failures.pop(label, None)
            LOGGER.info("Fetched %d rows for label '%s'.", rows, label)
        else:
            failures[label] = error
            LOGGER.error("Failed to fetch label '%s': %s", label, error)

    if failures:
        LOGGER.info("Retrying %d labels with extended spacing (60-90s).", len(failures))
        retry_queue = list(failures.items())
        failures.clear()
        for idx, (label, prev_error) in enumerate(retry_queue, start=1):
            key = (label, timeframe, geo)
            if key in completed_keys:
                continue
            queries = label_queries.get(label)
            if not queries:
                failures[label] = prev_error or "missing queries"
                continue
            polite_sleep(60.0, jitter=0.50)  # ensures 60-90s delay before retry
            success, error, rows = attempt_label(
                label,
                queries,
                timeframe,
                geo,
                args,
                base_delay,
                backoff_cap,
                max_non429,
                start_dt,
                source_batch,
                checkpoint_fields,
                local_cache,
                use_cache=False,
            )
            if success:
                completed_keys.add(key)
                successes += 1
                LOGGER.info("Retry succeeded for label '%s' (%d rows).", label, rows)
            else:
                failures[label] = error or prev_error
                LOGGER.error("Retry failed for label '%s': %s", label, failures[label])

    emit_coverage_report(total_labels, pre_completed, successes, failures, args.coverage_report)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
