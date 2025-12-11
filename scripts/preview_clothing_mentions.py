#!/usr/bin/env python3
"""
Quick helper to preview fields emitted by detect_clothing_mentions.py without writing Parquet.

Example:
  python preview_clothing_mentions.py \
      --in data/cur/lyrics.parquet \
      --taxonomy configs/taxonomy_v2.json \
      --aliases configs/aliases_v1.json \
      --field canonical_label \
      --max 20
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from detect_clothing_mentions import (
    compile_patterns,
    get_context_window,
    load_aliases,
    load_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a single field emitted by detect_clothing_mentions.py."
    )
    parser.add_argument("--in", "--lyrics", dest="inp", required=True, help="Input lyrics parquet")
    parser.add_argument("--taxonomy", required=True, help="Taxonomy JSON (same as detect script)")
    parser.add_argument("--aliases", required=True, help="Alias JSON (same as detect script)")
    parser.add_argument("--field", default=None, help="Single field to print (deprecated by --fields).")
    parser.add_argument(
        "--fields",
        default="canonical_label",
        help="Comma-separated list of fields to print (default: canonical_label).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Maximum number of mentions to print (default: 20)",
    )
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=20,
        help="Context window token span (mirrors detect_clothing_mentions; default: 20)",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror detect_clothing_mentions defaults for missing columns."""
    df = df.copy()
    if "song_id" not in df.columns:
        df["song_id"] = None
    if "release_date" not in df.columns:
        df["release_date"] = None
    if "source" not in df.columns:
        df["source"] = "genius"
    if "genius_url" not in df.columns:
        df["genius_url"] = None
    if "pageviews" not in df.columns:
        df["pageviews"] = None
    if "hot" not in df.columns:
        df["hot"] = False
    return df


def iter_mentions(
    df: pd.DataFrame,
    patterns,
    window_tokens: int,
) -> Iterable[Dict]:
    """Generator that yields mention dicts using the same logic as detect_clothing_mentions."""
    required = {"song_title", "artist", "lyric_text"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input parquet missing columns: {missing}")

    for _, row in df.iterrows():
        text = (row["lyric_text"] or "").replace("\r", "")
        if not text.strip():
            continue

        lines = [ln for ln in text.split("\n") if ln.strip()]
        for line_index, line in enumerate(lines):
            matches: List[tuple] = []
            for rx, canon, typ in patterns:
                for m in rx.finditer(line):
                    matches.append((m.start(), m.end(), m.group(0), canon, typ))
            if not matches:
                continue

            matches.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
            occupied = [False] * (len(line) + 1)
            for start, end, surface, canon, mention_type in matches:
                if any(occupied[start:end]):
                    continue
                for pos in range(start, end):
                    occupied[pos] = True
                context = get_context_window(line, start, end, window_tokens)
                yield {
                    "song_id": row.get("song_id"),
                    "line_id": line_index,
                    "mention_type": mention_type,
                    "surface_form": surface,
                    "canonical_label": canon,
                    "start_char": int(start),
                    "end_char": int(end),
                    "context_window": context,
                    "title": row["song_title"],
                    "artist": row["artist"],
                    "release_date": row.get("release_date"),
                    "source": row.get("source", "genius"),
                    "genius_url": row.get("genius_url"),
                    "pageviews": row.get("pageviews"),
                    "hot": bool(row.get("hot", False)),
                }


def main() -> int:
    args = parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    alias_map = load_aliases(args.aliases)
    patterns = compile_patterns(taxonomy, alias_map)

    lyrics_df = pd.read_parquet(args.inp)
    lyrics_df = ensure_columns(lyrics_df)

    if args.field and args.fields and args.field not in args.fields.split(","):
        args.fields = args.field

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        fields = ["canonical_label"]
    max_rows = max(1, args.max)

    printed = 0
    for mention in iter_mentions(lyrics_df, patterns, args.window_tokens):
        line = []
        for field in fields:
            if field not in mention:
                raise KeyError(f"Field '{field}' not found in mention: {sorted(mention.keys())}")
            value = mention[field]
            line.append(f"{field}={value!r}")
        print(" | ".join(line))
        printed += 1
        if printed >= max_rows:
            break

    if printed == 0:
        print("No mentions found; nothing to preview.")
    else:
        print(f"Printed {printed} mention rows with fields: {', '.join(fields)}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
