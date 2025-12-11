#!/usr/bin/env python3
"""
Detect clothing/material/accessory mentions in lyrics and write a mentions Parquet.

Input (Parquet): data/cur/lyrics.parquet
  columns: song_id, song_title, artist, release_date, source, lyric_text

Output (Parquet): data/cur/clothing_mentions.parquet
  columns:
    mention_id, song_id, line_id, mention_type, surface_form, canonical_label,
    detector, confidence, start_char, end_char, context_window,
    sentiment_label, sentiment_score, title, artist, release_date, source

Usage example:
  python detect_clothing_mentions.py \
    --in data/cur/lyrics.parquet \
    --out data/cur/clothing_mentions.parquet \
    --taxonomy configs/taxonomy_v1.json \
    --aliases configs/aliases_v1.json \
    --window-tokens 20
"""
from __future__ import annotations
import argparse, json, os, re, uuid
from typing import Dict, List, Tuple
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", "--lyrics", dest="inp", required=True,
                   help="Input parquet with lyrics (alias: --lyrics)")
    p.add_argument("--out", dest="outp", required=True)
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--aliases", required=True)
    p.add_argument("--window-tokens", type=int, default=20)
    return p.parse_args()

def load_taxonomy(tax_path: str) -> List[Dict]:
    with open(tax_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize
    norm = []
    for it in data:
        canon = it["canonical_label"].strip()
        typ = it["type"].strip().lower()  # clothing|material|accessory
        aliases = [a.strip() for a in it.get("aliases", []) if a.strip()]
        norm.append({"canonical_label": canon, "type": typ, "aliases": aliases})
    return norm

def load_aliases(alias_path: str) -> Dict[str, str]:
    with open(alias_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # {alias: canonical}
    return {k.strip(): v.strip() for k, v in data.items() if k.strip() and v.strip()}

def compile_patterns(tax: List[Dict], alias_map: Dict[str, str]) -> List[Tuple[re.Pattern, str, str]]:
    """
    Return list of (regex, canonical, type).
    We include:
      - each canonical label as a pattern
      - each taxonomy alias
      - each extra alias from alias_map
    Longer phrases matched first (we’ll sort by length).
    """
    items = []  # (phrase, canonical, type)
    # from taxonomy
    for it in tax:
        canon = it["canonical_label"]
        typ = it["type"]
        items.append((canon, canon, typ))
        for al in it.get("aliases", []):
            items.append((al, canon, typ))
    # from alias dictionary (type unknown -> infer from taxonomy by canonical; if missing, default clothing)
    canon2type = {it["canonical_label"]: it["type"] for it in tax}
    for al, canon in alias_map.items():
        typ = canon2type.get(canon, "clothing")
        items.append((al, canon, typ))

    # sort by length desc so multi-word/longer phrases win
    items.sort(key=lambda x: len(x[0]), reverse=True)

    patterns: List[Tuple[re.Pattern, str, str]] = []
    for phrase, canon, typ in items:
        # Build a word-boundary regex; support spaces as \s+; match case-insensitive
        safe = re.escape(phrase)
        safe = safe.replace(r"\ ", r"\s+")
        # \b is fine for most words; also guard apostrophes (’)
        rx = re.compile(rf"(?<!\w){safe}(?!\w)", flags=re.IGNORECASE)
        patterns.append((rx, canon, typ))
    return patterns

def get_context_window(line: str, start: int, end: int, window_tokens: int) -> str:
    # simple token-based window within the same line
    pre = line[:start]
    mid = line[start:end]
    post = line[end:]

    def tok_split(s: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)

    pre_toks = tok_split(pre)
    post_toks = tok_split(post)
    ctx = " ".join(pre_toks[-window_tokens:]) + (" " if pre_toks else "") + mid + (" " if post_toks else "") + " ".join(post_toks[:window_tokens])
    return ctx.strip()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.outp), exist_ok=True)

    tax = load_taxonomy(args.taxonomy)
    alias_map = load_aliases(args.aliases)
    patterns = compile_patterns(tax, alias_map)

    df = pd.read_parquet(args.inp)
    cols_needed = {"song_title","artist","lyric_text"}
    if not cols_needed.issubset(df.columns):
        raise SystemExit(f"Input parquet missing columns: {cols_needed - set(df.columns)}")

    # be robust if ids/dates/source are missing
    if "song_id" not in df.columns: df["song_id"] = None
    if "release_date" not in df.columns: df["release_date"] = None
    if "source" not in df.columns: df["source"] = "genius"
    if "genius_url" not in df.columns: df["genius_url"] = None
    if "pageviews" not in df.columns: df["pageviews"] = None
    if "hot" not in df.columns: df["hot"] = False

    out_rows = []
    mention_counter = 0

    for idx, row in df.iterrows():
        song_id = row.get("song_id")
        title = row["song_title"]
        artist = row["artist"]
        release_date = row.get("release_date")
        source = row.get("source", "genius")
        genius_url = row.get("genius_url")
        pageviews = row.get("pageviews")
        hot = bool(row.get("hot", False))
        text = (row["lyric_text"] or "").replace("\r", "")

        if not text.strip():
            continue

        # process line-by-line
        lines = [ln for ln in text.split("\n") if ln.strip()]
        for li, line in enumerate(lines):
            # run all patterns; de-duplicate overlapping by taking longest spans first
            matches = []
            for rx, canon, typ in patterns:
                for m in rx.finditer(line):
                    matches.append((m.start(), m.end(), m.group(0), canon, typ))
            if not matches:
                continue
            # sort: longer span first, then earlier start
            matches.sort(key=lambda t: (-(t[1]-t[0]), t[0]))
            taken = []
            occupied = [False]* (len(line)+1)

            for s, e, surf, canon, typ in matches:
                if any(occupied[s:e]):  # overlaps something longer already taken
                    continue
                for k in range(s, e):
                    occupied[k] = True
                taken.append((s,e,surf,canon,typ))

            for s,e,surf,canon,typ in taken:
                mention_counter += 1
                context = get_context_window(line, s, e, args.window_tokens)
                out_rows.append({
                    "mention_id": str(uuid.uuid4()),
                    "song_id": song_id,
                    "line_id": li,
                    "mention_type": typ,  # clothing|material|accessory
                    "surface_form": surf,
                    "canonical_label": canon,
                    "detector": "keyword",
                    "confidence": 1.0,
                    "start_char": int(s),
                    "end_char": int(e),
                    "context_window": context,
                    "sentiment_label": "neu",
                    "sentiment_score": 0.0,
                    "title": title,
                    "artist": artist,
                    "release_date": release_date,
                    "source": source,
                    "genius_url": genius_url,
                    "pageviews": pageviews,
                    "hot": hot,
                })

    if not out_rows:
        print("No mentions found. Check taxonomy/aliases or sample lyrics.")
        pd.DataFrame([], columns=[
            "mention_id","song_id","line_id","mention_type","surface_form","canonical_label",
            "detector","confidence","start_char","end_char","context_window","sentiment_label",
            "sentiment_score","title","artist","release_date","source","genius_url","pageviews","hot"
        ]).to_parquet(args.outp, index=False)
        return

    mdf = pd.DataFrame(out_rows)
    for col in ["pageviews"]:
        if col in mdf.columns:
            mdf[col] = pd.to_numeric(mdf[col], errors="coerce")
    if "hot" in mdf.columns:
        mdf["hot"] = mdf["hot"].fillna(False).astype(bool)
    mdf.to_parquet(args.outp, index=False)
    print(f"Saved {len(mdf)} mentions -> {args.outp}")

if __name__ == "__main__":
    main()
