# HypeBeats RAG System - SQL Aggregation Fix Implementation Plan

## Problem Statement
Count/ranking queries (e.g., "Which artists have most diverse brand vocabulary?") return wrong results because:
1. Semantic search returns only 30 rows, counts are computed from this subset
2. Example: A$AP Rocky has 15 unique brands in DB, but system reports 2

## Solution Overview
Add SQL aggregation path for count/frequency queries that bypasses semantic search and queries the full database directly.

---

## Files to Modify

| File | Action |
|------|--------|
| `app/services/trends_service.py` | Add `use_sql_aggregation` field to TrendDecision |
| `app/services/sql_aggregation.py` | CREATE NEW - SQL aggregation functions |
| `app/query_rag.py` | Update decision prompt, add routing, add deduplication |
| `app/services/synthesizer.py` | Add aggregation_data parameter |

---

## STEP 1: Update TrendDecision Model

**File:** `app/services/trends_service.py`

**Find this class (around line 719):**
```python
class TrendDecision(BaseModel):
    needs_trends: bool
    brand: str | None
    artist_names: list[str] = Field(default_factory=list)
    start_date: str | None
    end_date: str | None
    comparative_query: bool = False
```

**Replace with:**
```python
class TrendDecision(BaseModel):
    needs_trends: bool
    use_sql_aggregation: bool = False  # True for count/frequency/ranking queries
    brand: str | None
    artist_names: list[str] = Field(default_factory=list)
    start_date: str | None
    end_date: str | None
    comparative_query: bool = False
```

---

## STEP 2: Create SQL Aggregation Module

**File:** `app/services/sql_aggregation.py` (CREATE NEW)

```python
"""SQL aggregation for accurate count/ranking queries."""
import pandas as pd
import logging
from sqlalchemy import text
from app.config.settings import get_settings
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class SQLAggregator:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(settings.database.service_url)

    def get_artist_brand_diversity(self, artist_name: str = None, limit: int = 50) -> pd.DataFrame:
        """Unique brand count per artist."""
        query = """
        SELECT metadata->>'artist_name' as artist,
               COUNT(DISTINCT metadata->>'brand_name') as unique_brands,
               ARRAY_AGG(DISTINCT metadata->>'brand_name') as brands_list
        FROM brand_mentions
        """ + (f"WHERE metadata->>'artist_name' ILIKE '%{artist_name}%'" if artist_name else "") + """
        GROUP BY metadata->>'artist_name'
        ORDER BY unique_brands DESC
        LIMIT :limit
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'limit': limit})

    def get_brand_mention_counts(self, limit: int = 50) -> pd.DataFrame:
        """Top brands by mentions and unique songs."""
        query = """
        SELECT metadata->>'brand_name' as brand,
               COUNT(*) as total_mentions,
               COUNT(DISTINCT metadata->>'song_title') as unique_songs,
               COUNT(DISTINCT metadata->>'artist_name') as unique_artists
        FROM brand_mentions
        WHERE metadata->>'brand_name' IS NOT NULL
        GROUP BY metadata->>'brand_name'
        ORDER BY total_mentions DESC
        LIMIT :limit
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'limit': limit})

    def get_song_brand_density(self, limit: int = 50) -> pd.DataFrame:
        """Songs with most brand references."""
        query = """
        SELECT metadata->>'song_title' as song,
               metadata->>'artist_name' as artist,
               metadata->>'release_date' as release_date,
               COUNT(DISTINCT metadata->>'brand_name') as unique_brands,
               ARRAY_AGG(DISTINCT metadata->>'brand_name') as brands_list
        FROM brand_mentions
        WHERE metadata->>'brand_name' IS NOT NULL
        GROUP BY metadata->>'song_title', metadata->>'artist_name', metadata->>'release_date'
        ORDER BY unique_brands DESC
        LIMIT :limit
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'limit': limit})

    def get_artist_top_brands(self, artist_name: str, limit: int = 20) -> pd.DataFrame:
        """Top brands for specific artist."""
        query = """
        SELECT metadata->>'brand_name' as brand,
               COUNT(*) as mention_count,
               COUNT(DISTINCT metadata->>'song_title') as unique_songs
        FROM brand_mentions
        WHERE metadata->>'artist_name' ILIKE :pattern
          AND metadata->>'brand_name' IS NOT NULL
        GROUP BY metadata->>'brand_name'
        ORDER BY mention_count DESC
        LIMIT :limit
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'pattern': f'%{artist_name}%', 'limit': limit})

    def get_brand_by_artist_category(self, limit: int = 50) -> pd.DataFrame:
        """Brands with artist associations for luxury/streetwear categorization."""
        query = """
        SELECT metadata->>'brand_name' as brand,
               ARRAY_AGG(DISTINCT metadata->>'artist_name') as artists,
               COUNT(*) as mention_count
        FROM brand_mentions
        WHERE metadata->>'brand_name' IS NOT NULL
        GROUP BY metadata->>'brand_name'
        ORDER BY mention_count DESC
        LIMIT :limit
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'limit': limit})


def deduplicate_brand_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (artist, song, brand) tuples."""
    if df.empty:
        return df
    cols = [c for c in ['artist_name', 'song_title', 'brand_name'] if c in df.columns]
    if cols:
        original = len(df)
        df = df.drop_duplicates(subset=cols, keep='first')
        if len(df) < original:
            logger.info(f"Deduplication: {original} -> {len(df)} rows")
    return df


def route_aggregation_query(query: str, decision, aggregator: SQLAggregator) -> dict:
    """Route to appropriate SQL function based on query intent."""
    q = query.lower()
    
    # Artist-specific brands
    if decision.artist_names and any(kw in q for kw in ['top brands', 'brands referenced', 'discography']):
        results = aggregator.get_artist_top_brands(decision.artist_names[0])
        return {'aggregation_results': results, 'aggregation_type': 'artist_top_brands',
                'summary': f"Top brands for {decision.artist_names[0]}"}
    
    # Diverse vocabulary
    if 'diverse' in q or 'vocabulary' in q:
        results = aggregator.get_artist_brand_diversity()
        return {'aggregation_results': results, 'aggregation_type': 'artist_brand_diversity',
                'summary': f"Artists ranked by brand diversity"}
    
    # Songs with most brands
    if 'songs' in q and any(kw in q for kw in ['highest', 'most', 'brand references']):
        results = aggregator.get_song_brand_density()
        return {'aggregation_results': results, 'aggregation_type': 'song_brand_density',
                'summary': f"Songs ranked by brand count"}
    
    # Luxury vs streetwear
    if 'luxury' in q or 'streetwear' in q:
        results = aggregator.get_brand_by_artist_category()
        return {'aggregation_results': results, 'aggregation_type': 'brand_by_artist_category',
                'summary': f"Brands with artist associations"}
    
    # Default: brand mention counts
    results = aggregator.get_brand_mention_counts()
    return {'aggregation_results': results, 'aggregation_type': 'brand_mention_counts',
            'summary': f"Brands ranked by mention count"}
```

---

## STEP 3: Update query_rag.py

### 3A: Add Import

**Find imports section (top of file), add:**
```python
from app.services.sql_aggregation import SQLAggregator, deduplicate_brand_mentions, route_aggregation_query
```

### 3B: Replace Decision Prompt

**Find the `messages=[{` block in `query_system()` function (around line 298-370).**

**Replace the entire system content string with:**
```python
            "content": """You classify queries for a RAG system analyzing hip-hop lyrics and fashion trends.

OUTPUT SCHEMA:
- needs_trends: bool (Google Trends temporal analysis?)
- use_sql_aggregation: bool (direct SQL counts from full database?)
- comparative_query: bool (comparing 2+ brands/artists?)
- brand: str|None (specific brand if single-brand query)
- artist_names: list[str] (artists mentioned)
- start_date/end_date: str|None (ISO format)

CLASSIFICATION:

TYPE 1-5: TREND ANALYSIS (needs_trends=True, use_sql_aggregation=False)
- Single brand performance: "Did Gucci spike after..."
- Comparative trends: "Compare Nike vs Adidas trends"
- Brand report: "Marketing report for Tom Ford"
- Trajectory: "Influence trajectory of Versace"

TYPE 6: COUNT/AGGREGATION (needs_trends=False, use_sql_aggregation=True)
Triggers: "how many", "count", "most", "top", "highest", "ranking", "diverse", "frequently", "common", "unique", "which brands", "which artists", "which songs"
Examples:
- "Which artists have most diverse brand vocabulary?" -> use_sql_aggregation=True
- "Which brands appeared in highest number of songs?" -> use_sql_aggregation=True
- "For Kendrick, what are the top brands?" -> use_sql_aggregation=True, artist_names=["Kendrick Lamar"]
- "Which brands mentioned by luxury vs streetwear artists?" -> use_sql_aggregation=True

TYPE 3: LISTING (needs_trends=False, use_sql_aggregation=False)
- "What fashion items does Future mention?" -> just semantic search

CRITICAL: COUNT queries (most/top/highest/diverse) = TYPE 6, NOT trends"""
```

### 3C: Add SQL Aggregation Routing

**Find line after `decision = client.chat.completions.create(...)` ends (after the closing `)`), before `vs = VectorStore()`.**

**Insert:**
```python
    # SQL aggregation for count/ranking queries (TYPE 6)
    aggregation_data = None
    if decision.use_sql_aggregation:
        print("📊 SQL Aggregation Query - Direct Database Counts")
        aggregator = SQLAggregator()
        aggregation_data = route_aggregation_query(question, decision, aggregator)
        print(f"  Type: {aggregation_data['aggregation_type']}")
        print(f"  Rows: {len(aggregation_data['aggregation_results'])}")
        print(aggregation_data['aggregation_results'].head(15).to_string(index=False))
```

### 3D: Disable Auto-Trigger for Aggregation Queries

**Find this block (around line 422):**
```python
        if (not decision.needs_trends and
            not actual_brands_df.empty and
```

**Change to:**
```python
        if (not decision.needs_trends and
            not decision.use_sql_aggregation and  # ADD THIS LINE
            not actual_brands_df.empty and
```

### 3E: Add Deduplication Before Synthesis

**Find `synth = FashionSynthesizer()` (around line 704).**

**Insert BEFORE it:**
```python
        # Deduplicate brand mentions
        actual_brands_df = deduplicate_brand_mentions(actual_brands_df)
```

### 3F: Pass Aggregation Data to Synthesizer

**Find the `insight = synth.generate_insight(...)` call.**

**Add `aggregation_data` as the last parameter:**
```python
        insight = synth.generate_insight(
            question,
            actual_brands_df,
            enriched_results,
            lyrics_results,
            taxonomy_results,
            trends_data_list,
            decision.comparative_query,
            None,
            fashion_trends,
            aggregation_data  # ADD THIS
        )
```

---

## STEP 4: Update Synthesizer

**File:** `app/services/synthesizer.py`

### 4A: Update Method Signature

**Find `def generate_insight(...)`, add parameter:**
```python
def generate_insight(
    self,
    question: str,
    brand_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    lyrics_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    trends_data: list[dict],
    comparative: bool = False,
    category_baseline: dict = None,
    fashion_trends: list = None,
    aggregation_data: dict = None  # ADD THIS
) -> FashionInsight:
```

### 4B: Add Aggregation Context to Prompt

**Inside `generate_insight()`, before building the main context, add:**
```python
        # SQL aggregation results (accurate counts)
        agg_context = ""
        if aggregation_data:
            agg_df = aggregation_data['aggregation_results']
            agg_context = f"""
=== SQL AGGREGATION (ACCURATE COUNTS FROM FULL DATABASE) ===
Type: {aggregation_data['aggregation_type']}
{aggregation_data['summary']}

{agg_df.head(30).to_string(index=False)}

IMPORTANT: Use these counts as PRIMARY source. They are accurate, not estimates.
"""
```

**Then prepend `agg_context` to the context passed to the LLM.**

---

## Testing

After implementation, test these queries:

```python
# Should use SQL aggregation (use_sql_aggregation=True, needs_trends=False)
"Which artists have the most diverse brand vocabulary?"
"Which brands have appeared in the highest number of unique songs?"
"For Kendrick Lamar, what are the top brands referenced?"

# Should still use trends (use_sql_aggregation=False, needs_trends=True)  
"Generate a marketing insight report for 'Tom Ford'"
"Did Gucci spike after Migos Culture album?"
```

Expected: A$AP Rocky should show 15 unique brands, not 2.

---

## Summary of Changes

| Change | Lines | Purpose |
|--------|-------|---------|
| TrendDecision field | 1 line | Add `use_sql_aggregation` flag |
| New sql_aggregation.py | ~100 lines | SQL functions for counts |
| Decision prompt | ~30 lines | Add TYPE 6 classification |
| SQL routing | ~8 lines | Route to SQL before semantic search |
| Auto-trigger check | 1 line | Skip trends for aggregation |
| Deduplication | 1 line | Remove duplicate rows |
| Synthesizer param | 2 lines | Pass SQL results to LLM |
