#!/usr/bin/env python3
"""
Fetch lyrics & metadata from Genius and save a clean parquet for downstream Clothing/Brand detectors.

Output schema (one row per song):
- song_id (int)
- song_title (str)
- artist (str)
- release_date (date, yyyy-mm-dd or None)
- source (str, fixed "genius")
- genius_url (str)
- pageviews (int or None)
- hot (bool)
- lyric_text_raw (str)
- lyric_text (str, cleaned)

Usage examples:
  export GENIUS_TOKEN=xxxxxxxxxxxxxxxx
  python hypebeats_fetch_lyrics.py --artists "Travis Scott" "Drake" --max-songs 50 --out data/cur/lyrics.parquet
  python hypebeats_fetch_lyrics.py --artists-csv configs/artists_top50.csv --max-songs 40 --since 2024-01-01 --out data/cur/lyrics.parquet

Requires:
  pip install pandas pyarrow requests python-dateutil beautifulsoup4 tqdm python-dotenv langid
Optional (preferred for lyrics): pip install lyricsgenius
"""
from __future__ import annotations
import os, sys, time, json, argparse, re
from typing import List, Dict, Optional, Iterable
import requests
import pandas as pd
from datetime import datetime
from dateutil import parser as dateparser
from bs4 import BeautifulSoup
from tqdm import tqdm
import langid

# Optional import: lyricsgenius (uses the same token; more robust for lyrics)
try:
    import lyricsgenius
    HAS_LYRICS_GENIUS = True
except Exception:
    HAS_LYRICS_GENIUS = False

GENIUS_API = "https://api.genius.com"
USER_AGENT = "hypebeats-lyrics-fetcher/1.0 (+https://localhost)"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch lyrics & metadata from Genius")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--artists", nargs="+", help="Artist names (one or more)")
    g.add_argument("--artists-csv", help="CSV with a column 'artist'")

    p.add_argument("--since", help="Only include songs released on/after this date (YYYY-MM-DD)")
    p.add_argument("--until", help="Only include songs released on/before this date (YYYY-MM-DD)")
    p.add_argument("--max-songs", type=int, default=50, help="Max songs per artist (default 50)")
    p.add_argument("--sleep-sec", type=float, default=0.6, help="Sleep between API calls to be polite")
    p.add_argument("--out", required=True, help="Path to parquet output")
    p.add_argument("--use-lyricsgenius", action="store_true", help="Use lyricsgenius if installed (recommended)")
    p.add_argument("--min-english-prob", type=float, default=0.90, help="Keep lyrics if langid predicts EN ≥ this prob")
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate songs by (artist,title)")
    p.add_argument("--min-pageviews", type=int, default=0, help="Keep songs with ≥ this many Genius pageviews")
    p.add_argument("--only-hot", action="store_true", help="Keep songs flagged as hot on Genius")
    p.add_argument("--dotenv", default=None, help="Optional path to a .env file")
    return p.parse_args()

def load_env(dotenv_path: Optional[str]) -> None:
    if dotenv_path:
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
        except Exception:
            pass

def get_token() -> str:
    token = os.getenv("GENIUS_TOKEN")
    if not token:
        raise SystemExit("Missing GENIUS_TOKEN. Create a Genius Client Access Token and export GENIUS_TOKEN.")
    return token

def date_from_components(obj: Dict) -> Optional[str]:
    comp = obj.get("release_date_components") or {}
    y, m, d = comp.get("year"), comp.get("month"), comp.get("day")
    if y and m and d:
        return f"{y:04d}-{m:02d}-{d:02d}"
    # fallback display field
    disp = obj.get("release_date_for_display")
    if disp:
        try:
            return dateparser.parse(disp).date().isoformat()
        except Exception:
            return None
    return None

def within_bounds(dt_str: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    if not dt_str:
        return True  # keep if unknown
    try:
        d = dateparser.parse(dt_str).date()
        if since and d < dateparser.parse(since).date(): return False
        if until and d > dateparser.parse(until).date(): return False
        return True
    except Exception:
        return True

def genius_search_songs(session: requests.Session, token: str, artist: str, max_songs: int, sleep_sec: float) -> List[Dict]:
    """Search and collect candidate songs for an artist using Genius search + filtering by primary artist."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    page = 1
    hits: List[Dict] = []
    while len(hits) < max_songs:
        r = session.get(f"{GENIUS_API}/search", params={"q": artist, "page": page}, headers=headers, timeout=20)
        if r.status_code != 200:
            break
        js = r.json().get("response", {}).get("hits", [])
        if not js: break
        for h in js:
            res = h.get("result", {})
            primary = res.get("primary_artist", {}).get("name", "")
            if primary.lower() != artist.lower():
                continue
            hits.append(res)
            if len(hits) >= max_songs:
                break
        page += 1
        time.sleep(sleep_sec)
    return hits

def get_song_detail(session: requests.Session, token: str, song_id: int) -> Optional[Dict]:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    r = session.get(f"{GENIUS_API}/songs/{song_id}", headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    return r.json().get("response", {}).get("song")

def extract_pageviews(detail: Dict) -> Optional[int]:
    stats = (detail or {}).get("stats") or {}
    pv = stats.get("pageviews")
    if pv is None:
        return None
    try:
        return int(pv)
    except (TypeError, ValueError):
        return None

def extract_hot(detail: Dict) -> bool:
    try:
        return bool(detail.get("hot", False))
    except Exception:
        return False

def scrape_lyrics_from_url(session: requests.Session, url: str) -> Optional[str]:
    """Very simple page scrape fallback (Genius may change markup; lyricsgenius is preferred)."""
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # Genius typically wraps lyrics in <div data-lyrics-container="true"> chunks
        blocks = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not blocks:
            # older pages may have .lyrics class
            lyr = soup.select_one(".lyrics")
            return lyr.get_text("\n").strip() if lyr else None
        text = "\n".join(b.get_text("\n") for b in blocks)
        # remove bracketed stage directions [Chorus], etc. (keep them if you want)
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        return text
    except Exception:
        return None

def fetch_lyrics(token: str, artist: str, max_songs: int, since: Optional[str], until: Optional[str], sleep_sec: float, use_lyricsgenius: bool) -> List[Dict]:
    session = requests.Session()
    out: List[Dict] = []

    if use_lyricsgenius and HAS_LYRICS_GENIUS:
        api = lyricsgenius.Genius(token, timeout=20, sleep_time=sleep_sec, remove_section_headers=False)
        api.skip_non_songs = True
        api.excluded_terms = ["(Remix)", "(Live)"]
        songs = api.search_artist(artist, max_songs=max_songs, sort="title")
        if not songs:
            return out
        for s in songs.songs:
            # lyricsgenius Song has limited metadata; we’ll pull detail endpoint for date
            song_id = s.id
            detail = get_song_detail(session, token, song_id)
            if not detail:
                continue
            release = date_from_components(detail)
            if not within_bounds(release, since, until):
                continue
            lyr = s.lyrics or ""
            if not lyr:
                # fallback scrape
                lyr = scrape_lyrics_from_url(session, detail.get("url", "")) or ""
            out.append({
                "song_id": int(song_id),
                "song_title": s.title,
                "artist": artist,
                "release_date": release,
                "source": "genius",
                "genius_url": detail.get("url"),
                "pageviews": extract_pageviews(detail),
                "hot": extract_hot(detail),
                "lyric_text": lyr,
            })
            time.sleep(sleep_sec)
        return out

    # Fallback: REST + scrape
    search_hits = genius_search_songs(session, token, artist, max_songs, sleep_sec)
    for res in search_hits:
        song_id = res.get("id")
        if song_id is None: 
            continue
        detail = get_song_detail(session, token, int(song_id))
        if not detail:
            continue
        release = date_from_components(detail)
        if not within_bounds(release, since, until):
            continue
        url = detail.get("url", "")
        lyr = scrape_lyrics_from_url(session, url) or ""
        out.append({
            "song_id": int(song_id),
            "song_title": detail.get("title", res.get("title", "")),
            "artist": artist,
            "release_date": release,
            "source": "genius",
            "genius_url": detail.get("url"),
            "pageviews": extract_pageviews(detail),
            "hot": extract_hot(detail),
            "lyric_text": lyr,
        })
        time.sleep(sleep_sec)
    return out

def load_artists(args: argparse.Namespace) -> List[str]:
    if args.artists:
        return args.artists
    df = pd.read_csv(args.artists_csv)
    if "artist" not in df.columns:
        raise SystemExit("artists CSV must have a column named 'artist'")
    return [str(a).strip() for a in df["artist"] if str(a).strip()]

def is_english(text: str, min_prob: float) -> bool:
    if not text.strip():
        return False
    lang, score = langid.classify(text)
    if lang != "en":
        return False
    # langid <=1.1 returns a probability, newer releases return a log-probability.
    if 0 <= score <= 1:
        return score >= min_prob
    # When score is outside [0,1], treat it as confidence in log space and accept.
    return True

def main():
    args = parse_args()
    load_env(args.dotenv)
    token = get_token()
    artists = load_artists(args)

    rows: List[Dict] = []
    for artist in artists:
        print(f"==> {artist}")
        try:
            rows.extend(fetch_lyrics(
                token=token,
                artist=artist,
                max_songs=args.max_songs,
                since=args.since,
                until=args.until,
                sleep_sec=args.sleep_sec,
                use_lyricsgenius=args.use_lyricsgenius
            ))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] {artist}: {e}")

    if not rows:
        print("No songs collected. Check token/artist names/date filters.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # Basic cleaning
    df["lyric_text_raw"] = df["lyric_text"]
    df["lyric_text"] = df["lyric_text"].fillna("").apply(lambda t: re.sub(r"\r\n?", "\n", t))
    df["pageviews"] = pd.to_numeric(df.get("pageviews"), errors="coerce").astype("Int64")
    df["hot"] = df.get("hot", False).fillna(False).astype(bool)
    if args.dedupe:
        df = df.sort_values(["artist", "song_title", "release_date"]).drop_duplicates(["artist","song_title"], keep="first")

    # Language filter (keep English)
    keep_mask = df["lyric_text"].apply(lambda t: is_english(t, args.min_english_prob))
    kept = int(keep_mask.sum())
    dropped = int((~keep_mask).sum())
    print(f"Lang filter: kept {kept}, dropped {dropped}")
    df = df[keep_mask].reset_index(drop=True)

    # Popularity filters
    if args.min_pageviews > 0:
        before = len(df)
        df = df[df["pageviews"].fillna(0) >= args.min_pageviews].reset_index(drop=True)
        print(f"Pageviews filter ({args.min_pageviews}+): kept {len(df)}, dropped {before - len(df)}")
    if args.only_hot:
        before = len(df)
        df = df[df["hot"]].reset_index(drop=True)
        print(f"Hot filter: kept {len(df)}, dropped {before - len(df)}")

    # Coerce datatypes & column order
    df["song_id"] = pd.to_numeric(df["song_id"], errors="coerce").astype("Int64")
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_date"] = df["release_date"].dt.date.apply(lambda d: d.isoformat() if pd.notna(d) else None)
    df["genius_url"] = df.get("genius_url").astype(str).replace({"nan": None})
    df["source"] = df.get("source", "genius").fillna("genius")

    # ensure expected columns exist even if API missed data
    cols = [
        "song_id",
        "song_title",
        "artist",
        "release_date",
        "source",
        "genius_url",
        "hot",
        "pageviews",
        "lyric_text_raw",
        "lyric_text",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith((".parquet", ".pq")):
        df.to_parquet(out_path, index=False)
    elif out_path.lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        raise SystemExit(f"Unsupported output extension for {out_path}. Use .parquet or .csv")

    print(f"Wrote {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    raise SystemExit(main())
