#!/usr/bin/env python3
"""
Seed a clothing-focused lyrics corpus by querying Genius with clothing/material/accessory terms
(and optional brand×garment combos), verifying matches appear in the lyrics, and writing a parquet.

Inputs:
  --queries-csv   CSV with column: query
  --brands-csv    Optional CSV with column: brand (e.g., "Versace","Gucci","Nike")
  --taxonomy      JSON taxonomy (to auto-derive canonical clothing terms)
  --aliases       JSON alias map (optional enrichment)
  --artists-csv   Optional CSV with column: artist (to constrain primary artists)
Outputs:
  data/cur/lyrics.parquet              (append/overwrite controlled by flags)
  data/cur/seed_matches.csv            (song_id, artist, title, matched_queries)

Env:
  GENIUS_TOKEN must be set; pip install lyricsgenius pandas pyarrow requests python-dateutil

Example:
  export GENIUS_TOKEN=...
  python hypebeats_build_clothing_corpus.py \
    --queries-csv configs/clothing_queries.csv \
    --brands-csv configs/top_brands.csv \
    --taxonomy configs/taxonomy_v1.json \
    --out data/cur/lyrics.parquet --audit-out data/cur/seed_matches.csv \
    --since 2024-01-01 --max-per-query 25
"""
from __future__ import annotations
import argparse, os, time, json, re
from typing import List, Dict, Optional, Set
import pandas as pd
import requests
from dateutil import parser as dparse

try:
    import lyricsgenius
    HAS_LG = True
except Exception:
    HAS_LG = False

GENIUS_API = "https://api.genius.com"

def parse_args():
    p = argparse.ArgumentParser("Build a clothing-focused lyrics corpus from Genius search.")
    p.add_argument("--queries-csv", help="CSV with column 'query'")
    p.add_argument("--brands-csv", help="CSV with column 'brand' (optional; used to create brand×garment queries)")
    p.add_argument("--taxonomy", required=True, help="taxonomy_v*.json with canonical clothing/material/accessory")
    p.add_argument("--aliases", default=None, help="aliases_v*.json (optional)")
    p.add_argument("--artists-csv", help="Optional CSV with column 'artist' to restrict to those primary artists")
    p.add_argument("--garments", nargs="*", default=["robe","dress","hoodie","jacket","coat","jeans","sneakers","flip flops","slides","boots"],
                   help="Garments list (used with brands to form 'brand + garment' queries)")
    p.add_argument("--since", default=None, help="Keep songs on/after this date (YYYY-MM-DD)")
    p.add_argument("--until", default=None, help="Keep songs on/before this date (YYYY-MM-DD)")
    p.add_argument("--max-per-query", type=int, default=25, help="Max songs to consider per query")
    p.add_argument("--sleep-sec", type=float, default=0.6, help="Sleep between API calls")
    p.add_argument("--out", required=True, help="Output parquet (lyrics).")
    p.add_argument("--audit-out", default="data/cur/seed_matches.csv", help="Where to save query→song matches.")
    p.add_argument("--append", action="store_true", help="Append to existing parquet if present.")
    return p.parse_args()

def load_list_csv(path: Optional[str], col: str) -> List[str]:
    if not path: return []
    df = pd.read_csv(path)
    if col not in df.columns:
        raise SystemExit(f"{path} must have a '{col}' column")
    return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]

def load_taxonomy(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_aliases(path: Optional[str]) -> Dict[str,str]:
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return {k.strip(): v.strip() for k,v in j.items() if k.strip() and v.strip()}

def canonical_clothing_terms(tax: List[Dict]) -> List[str]:
    out = []
    for it in tax:
        if it.get("type") in {"clothing","accessory","material"}:
            out.append(it["canonical_label"])
            for a in it.get("aliases", []):
                out.append(a)
    # de-dupe, preserve order
    seen = set(); uniq = []
    for t in out:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key); uniq.append(t)
    return uniq

def generate_queries(tax: List[Dict], aliases: Dict[str,str], garments: List[str], brands: List[str], queries_csv: Optional[str]) -> List[str]:
    base = []
    if queries_csv:
        base = load_list_csv(queries_csv, "query")
    # taxonomy terms as queries
    base += canonical_clothing_terms(tax)
    # brand × garment combos (e.g., "versace robe", "uggs boots", "gucci dress")
    for b in brands:
        for g in garments:
            base.append(f"{b} {g}")
    # normalize and dedupe
    seen = set(); final = []
    for q in base:
        qq = " ".join(str(q).strip().split())
        lq = qq.lower()
        if not qq or lq in seen: continue
        seen.add(lq); final.append(qq)
    return final

def get_token() -> str:
    import os
    tok = os.getenv("GENIUS_TOKEN")
    if not tok:
        raise SystemExit("GENIUS_TOKEN is not set.")
    return tok

def genius_headers(tok: str) -> Dict[str,str]:
    return {"Authorization": f"Bearer {tok}", "User-Agent": "hypebeats-clothing-corpus/1.0"}

def is_in_date_bounds(dt: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    if not dt: return True
    try:
        d = dparse.parse(dt).date()
        if since and d < dparse.parse(since).date(): return False
        if until and d > dparse.parse(until).date(): return False
        return True
    except Exception:
        return True

def get_song_detail(tok: str, song_id: int) -> Optional[Dict]:
    r = requests.get(f"{GENIUS_API}/songs/{song_id}", headers=genius_headers(tok), timeout=20)
    if r.status_code != 200: return None
    return r.json().get("response", {}).get("song")

def get_release_date(song: Dict) -> Optional[str]:
    comp = song.get("release_date_components") or {}
    y, m, d = comp.get("year"), comp.get("month"), comp.get("day")
    if y and m and d: return f"{y:04d}-{m:02d}-{d:02d}"
    disp = song.get("release_date_for_display")
    if disp:
        try: return dparse.parse(disp).date().isoformat()
        except Exception: return None
    return None

def search_hits(tok: str, query: str, max_songs: int, artist_whitelist: Optional[Set[str]], sleep_sec: float) -> List[Dict]:
    hits = []
    page = 1
    while len(hits) < max_songs:
        r = requests.get(f"{GENIUS_API}/search", params={"q": query, "page": page}, headers=genius_headers(tok), timeout=20)
        if r.status_code != 200: break
        arr = r.json().get("response", {}).get("hits", [])
        if not arr: break
        for h in arr:
            res = h.get("result", {})
            pa = (res.get("primary_artist", {}) or {}).get("name","")
            if artist_whitelist and pa.lower().strip() not in artist_whitelist:
                continue
            hits.append(res)
            if len(hits) >= max_songs: break
        page += 1
        time.sleep(sleep_sec)
    return hits

def scrape_or_api_lyrics(tok: str, song_detail: Dict) -> str:
    # Prefer lyricsgenius if present; else do a very simple page scrape best-effort
    url = song_detail.get("url", "")
    text = ""
    if HAS_LG:
        try:
            api = lyricsgenius.Genius(tok, timeout=20, sleep_time=0.5, remove_section_headers=False)
            s = api.search_song(title=song_detail.get("title",""), artist=song_detail.get("primary_artist",{}).get("name",""))
            if s and getattr(s, "lyrics", ""):
                text = s.lyrics
        except Exception:
            text = ""
    if not text:
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(url, headers={"User-Agent":"hypebeats-scraper/1.0"}, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            blocks = soup.find_all("div", attrs={"data-lyrics-container":"true"})
            if blocks:
                text = "\n".join(b.get_text("\n") for b in blocks)
            else:
                lyr = soup.select_one(".lyrics")
                text = lyr.get_text("\n").strip() if lyr else ""
        except Exception:
            text = ""
    return (text or "").replace("\r","")

def lyric_contains_term(text: str, query: str) -> bool:
    # Accept small variations of whitespace/case; if query is two words, enforce order
    q = re.escape(query).replace(r"\ ", r"\s+")
    rx = re.compile(rf"(?i)(?<!\w){q}(?!\w)")
    return bool(rx.search(text))

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.audit_out), exist_ok=True)

    tok = get_token()
    tax = load_taxonomy(args.taxonomy)
    aliases = load_aliases(args.aliases)
    brands = [b.lower() for b in load_list_csv(args.brands_csv, "brand")]
    artists = set(a.lower().strip() for a in load_list_csv(args.artists_csv, "artist")) if args.artists_csv else None

    queries = generate_queries(tax, aliases, args.garments, brands, args.queries_csv)
    print(f"Total queries: {len(queries)}")

    # Optional append existing
    existing = None
    if args.append and os.path.exists(args.out):
        try:
            existing = pd.read_parquet(args.out)
        except Exception:
            existing = None

    seen_key = set()
    if existing is not None:
        for _, r in existing.iterrows():
            k = (str(r.get("artist","")).lower().strip(), str(r.get("song_title","")).lower().strip())
            seen_key.add(k)

    rows = []
    audit = {}  # (artist,title) -> set(queries)

    for q in queries:
        hits = search_hits(tok, q, args.max_per_query, artists, args.sleep_sec)
        # Pull details + lyrics; keep only if lyrics actually contain the query
        for res in hits:
            sid = res.get("id")
            if sid is None: continue
            detail = get_song_detail(tok, int(sid))
            if not detail: continue
            pa = detail.get("primary_artist", {}).get("name","")
            title = detail.get("title","")
            if artists and pa.lower().strip() not in artists:
                continue
            release = get_release_date(detail)
            if not is_in_date_bounds(release, args.since, args.until):
                continue
            lyr = scrape_or_api_lyrics(tok, detail)
            if not lyr or not lyric_contains_term(lyr, q):
                continue
            k = (pa.lower().strip(), title.lower().strip())
            if k in seen_key:  # already have it
                audit.setdefault(k, set()).add(q)
                continue
            rows.append({
                "song_id": int(sid),
                "song_title": title,
                "artist": pa,
                "release_date": release,
                "source": "genius",
                "lyric_text": lyr
            })
            seen_key.add(k)
            audit.setdefault(k, set()).add(q)
            print(f"+ {pa} :: {title}   [q='{q}']")

        time.sleep(args.sleep_sec)

    if not rows and existing is None:
        print("No new rows gathered.")
        return

    out_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if existing is not None else pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved {len(out_df)} songs -> {args.out}")

    audit_rows = []
    for (a,t), qs in audit.items():
        # Find song_id if present
        sid = None
        m = out_df[(out_df["artist"].str.lower()==a) & (out_df["song_title"].str.lower()==t)]
        if len(m):
            sid = m.iloc[0].get("song_id")
        audit_rows.append({"artist": a, "title": t, "song_id": sid, "matched_queries": "; ".join(sorted(qs))})
    pd.DataFrame(audit_rows).to_csv(args.audit_out, index=False)
    print(f"Saved audit -> {args.audit_out}")

if __name__ == "__main__":
    main()
