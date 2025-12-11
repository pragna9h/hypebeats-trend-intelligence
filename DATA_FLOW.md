# HYPEBEATS RAG System - Complete Data Flow Walkthrough

> Follow a real query through the entire system with actual code snippets and data transformations

---

## Overview

This document traces **one complete query** through the RAG system, showing:
- Code executed at each step
- Data transformations
- SQL queries run
- API calls made
- LLM prompts and responses

**Example Query:** `"Did Gucci spike after Migos Culture album in January 2017?"`

---

## Table of Contents

1. [Initial Query Entry](#1-initial-query-entry)
2. [Date Extraction](#2-date-extraction)
3. [Query Classification](#3-query-classification)
4. [Embedding Generation](#4-embedding-generation)
5. [Parallel Vector Search](#5-parallel-vector-search)
6. [Brand/Item Separation](#6-branditem-separation)
7. [Trends Analysis](#7-trends-analysis)
8. [Context Formatting](#8-context-formatting)
9. [LLM Synthesis](#9-llm-synthesis)
10. [Output Formatting](#10-output-formatting)

---

## 1. Initial Query Entry

**File:** `rag-system/app/query_rag.py:738-843`

**Code:**
```python
if __name__ == "__main__":
    query_system("Did Gucci spike after Migos Culture album in January 2017?")
```

**Function Called:**
```python
def query_system(question: str, limit: int = 50):
    print(f"\n{'='*80}")
    print(f"Query: {question}")
    print(f"{'='*80}\n")

    # Start processing...
```

**Console Output:**
```
================================================================================
Query: Did Gucci spike after Migos Culture album in January 2017?
================================================================================
```

---

## 2. Date Extraction

**File:** `rag-system/app/query_rag.py:200-249`

**Code Executed:**
```python
start_date, end_date = extract_date_range(question)
```

**Function Implementation:**
```python
def extract_date_range(query: str) -> tuple[str, str]:
    """Extract start/end dates from query text."""

    # Handle "January 2017" (month-year)
    month_year = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        query,
        re.IGNORECASE
    )
    if month_year:
        month_name, year = month_year.groups()  # ("January", "2017")
        month_num = datetime.strptime(month_name, "%B").month  # 1
        last_day = calendar.monthrange(int(year), month_num)[1]  # 31
        return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day}"
        # Returns: ("2017-01-01", "2017-01-31")

    # ... other patterns ...
    return None, None
```

**Input:**
```
"Did Gucci spike after Migos Culture album in January 2017?"
```

**Regex Match:**
```python
match = re.search(r'(January|...|December)\s+(\d{4})', query, re.IGNORECASE)
# match.groups() = ("January", "2017")
```

**Output:**
```python
start_date = "2017-01-01"
end_date = "2017-01-31"
```

**Console Output:**
```
📅 Date Range: 2017-01-01 to 2017-01-31
```

---

## 3. Query Classification

**File:** `rag-system/app/query_rag.py:298-338`

**Code:**
```python
client = instructor.from_openai(OpenAI())
decision = client.chat.completions.create(
    model="gpt-5",
    response_model=TrendDecision,
    messages=[{
        "role": "system",
        "content": """You classify queries for a RAG system analyzing hip-hop lyrics and fashion trends.

OUTPUT SCHEMA:
- needs_trends: bool (Google Trends temporal analysis?)
- use_sql_aggregation: bool (direct SQL counts?)
- comparative_query: bool (comparing 2+ brands/artists?)
- brand: str|None (specific brand if single-brand query)
- artist_names: list[str] (artists mentioned)
- start_date/end_date: str|None (ISO format)

CLASSIFICATION:
TYPE 1-5: TREND ANALYSIS (needs_trends=True)
- Single brand performance: "Did Gucci spike after..."
- Comparative trends: "Compare Nike vs Adidas"

TYPE 6: COUNT/AGGREGATION (use_sql_aggregation=True)
Triggers: "how many", "top", "most", "ranking"
Examples: "Which artists have most diverse brand vocabulary?"
"""
    }, {
        "role": "user",
        "content": "Did Gucci spike after Migos Culture album in January 2017?"
    }]
)
```

**GPT-5 Response (JSON):**
```json
{
    "needs_trends": true,
    "use_sql_aggregation": false,
    "comparative_query": false,
    "brand": "Gucci",
    "artist_names": ["Migos"],
    "start_date": "2017-01-01",
    "end_date": "2017-01-31"
}
```

**Pydantic Validation:**
```python
class TrendDecision(BaseModel):
    needs_trends: bool
    use_sql_aggregation: bool
    comparative_query: bool
    brand: str | None
    artist_names: list[str]
    start_date: str | None
    end_date: str | None
```

**Validated Object:**
```python
decision = TrendDecision(
    needs_trends=True,           # ✅ Asking about spike/trend
    use_sql_aggregation=False,   # ✅ Not a count query
    comparative_query=False,     # ✅ Single brand
    brand="Gucci",               # ✅ Extracted
    artist_names=["Migos"],      # ✅ Extracted
    start_date="2017-01-01",     # ✅ From date extraction
    end_date="2017-01-31"        # ✅ From date extraction
)
```

---

## 4. Embedding Generation

**File:** `rag-system/app/database/vector_store.py:29-43`

**Code:**
```python
query_embedding = vs.get_embedding(question)
```

**Function Implementation:**
```python
def get_embedding(self, text: str) -> List[float]:
    """Generate embedding for text."""
    text = text.replace("\n", " ")

    start_time = time.time()

    embedding = (
        self.openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"  # 1536 dimensions
        )
        .data[0]
        .embedding
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Embedding generated in {elapsed_time:.3f}s")

    return embedding
```

**API Request:**
```http
POST https://api.openai.com/v1/embeddings
Content-Type: application/json
Authorization: Bearer sk-...

{
    "input": ["Did Gucci spike after Migos Culture album in January 2017?"],
    "model": "text-embedding-3-small"
}
```

**API Response:**
```json
{
    "object": "list",
    "data": [{
        "object": "embedding",
        "index": 0,
        "embedding": [
            0.023456,
            -0.145678,
            0.067890,
            0.890123,
            ...
            0.456789
        ]
    }],
    "model": "text-embedding-3-small",
    "usage": {
        "prompt_tokens": 14,
        "total_tokens": 14
    }
}
```

**Output:**
```python
query_embedding = [
    0.023456,
    -0.145678,
    0.067890,
    ...
    0.456789
]  # Length: 1536 floats
```

**Console Output:**
```
Embedding generated in 0.142s
```

---

## 5. Parallel Vector Search

**File:** `rag-system/app/query_rag.py:146-177`

**Code:**
```python
# Extract artist from database
artist_filter = extract_artist_filter(question, vs.conn)
# Returns: "Migos"

# Run 4 searches in parallel
results_dict = asyncio.run(
    parallel_search(vs, question, artist_filter, start_date, end_date, filter_brands=True)
)
```

**Function Implementation:**
```python
async def parallel_search(vs, question, artist_filter, start_date, end_date, filter_brands):
    """Execute 4 searches concurrently using thread pool."""
    loop = asyncio.get_event_loop()

    brand_artist_filter = artist_filter if filter_brands else None

    tasks = [
        # Task 1: Brand mentions with JOINs
        loop.run_in_executor(
            None,
            vs.search_with_joins,
            question, 30, start_date, end_date, True, brand_artist_filter
        ),

        # Task 2: Enriched lyrics (fashion items)
        loop.run_in_executor(
            None,
            vs.search,
            question, 40, None, True, "enriched_lyrics", artist_filter
        ),

        # Task 3: Full lyrics (song context)
        loop.run_in_executor(
            None,
            vs.search,
            question, 20, None, True, "full_lyrics", artist_filter
        ),

        # Task 4: Taxonomy (category baselines)
        loop.run_in_executor(
            None,
            vs.search_taxonomy,
            question, 5, True
        )
    ]

    # Wait for all 4 to complete
    results = await asyncio.gather(*tasks)

    return {
        'brand_mentions': results[0],
        'enriched_lyrics': results[1],
        'full_lyrics': results[2],
        'taxonomy': results[3]
    }
```

### Search 1: Brand Mentions with JOINs

**File:** `rag-system/app/database/vector_store.py:178-370`

**SQL Query:**
```sql
SELECT
    bm.id::text as id,
    bm.contents,
    'brand_mention' as source,
    s.song_title,
    s.release_date,
    a.artist_name,
    a.genre,
    a.region,
    b.brand_name,
    b.category,
    COALESCE((el_match.metadata->>'popularity_weight')::float, NULL) as popularity_weight,
    1 - (bm.embedding <=> $1::vector) as similarity
FROM brand_mentions bm
LEFT JOIN songs s ON bm.song_id = s.song_id
LEFT JOIN artists a ON s.artist_id = a.artist_id
LEFT JOIN brands b ON bm.brand_id = b.brand_id
LEFT JOIN enriched_lyrics el_match ON (
    LOWER(TRIM(el_match.metadata->>'artist')) = LOWER(TRIM(a.artist_name))
    AND LOWER(TRIM(el_match.metadata->>'title')) = LOWER(TRIM(s.song_title))
)
WHERE (
    (s.release_date ~ '^\\d{1,2}/\\d{1,2}/\\d{4}$' AND
     TO_DATE(s.release_date, 'FMMM/FMDD/YYYY') >= '2017-01-01'::date) OR
    (s.release_date ~ '^\\d{4}$' AND s.release_date::int >= 2017)
)
AND (
    (s.release_date ~ '^\\d{1,2}/\\d{1,2}/\\d{4}$' AND
     TO_DATE(s.release_date, 'FMMM/FMDD/YYYY') <= '2017-01-31'::date) OR
    (s.release_date ~ '^\\d{4}$' AND s.release_date::int <= 2017)
)
AND LOWER(a.artist_name) = 'migos'
ORDER BY similarity DESC
LIMIT 30
```

**Query Parameters:**
```python
params = [
    query_embedding,  # [0.023, -0.145, ...]
    # Date parameters inserted above
]
```

**Results (Top 3):**
| similarity | artist | song_title | date | brand_name | category | contents |
|------------|--------|-----------|------|----------|---------|----------|
| 0.9234 | Migos | T-Shirt | 1/15/2017 | Gucci | Luxury | Artist: Migos\nSong: T-Shirt\nBrand: Gucci\nContext: Gucci my t-shirt, Versace my shoes... |
| 0.8967 | Migos | Bad and Boujee | 1/27/2017 | Gucci | Luxury | Artist: Migos\nSong: Bad and Boujee\nBrand: Gucci\nContext: rain drop, drop top, smoking on cookie... |
| 0.8723 | Migos | Slippery | 1/27/2017 | Gucci | Luxury | Artist: Migos\nSong: Slippery\nBrand: Gucci\nContext: Gucci flip flops, fuck it, hit your bitch... |

**Time:** ~150ms

### Search 2: Enriched Lyrics

**SQL Query:**
```sql
SELECT
    el.id::text as id,
    el.contents,
    'enriched_lyrics' as source,
    el.metadata->>'title' as song_title,
    el.metadata->>'release_date' as release_date,
    el.metadata->>'artist' as artist_name,
    el.metadata->>'canonical_label' as brand_name,
    el.metadata->>'surface_form' as category,
    (el.metadata->>'popularity_weight')::float as popularity_weight,
    1 - (el.embedding <=> $1::vector) as similarity
FROM enriched_lyrics el
WHERE LOWER(el.metadata->>'artist') = 'migos'
ORDER BY similarity DESC
LIMIT 40
```

**Results (Top 3):**
| similarity | artist_name | song_title | canonical_label | surface_form | popularity_weight |
|------------|-------------|-----------|----------------|--------------|------------------|
| 0.9145 | Migos | T-Shirt | Gucci | Guwop | 87.5 |
| 0.8834 | Migos | Bad and Boujee | Gucci | Gucci | 95.2 |
| 0.8621 | Migos | Slippery | Versace | Versace | 82.1 |

**Time:** ~100ms

### Search 3: Full Lyrics

**Results:** 20 full song lyrics from Migos (truncated to 500 chars each)

**Time:** ~120ms

### Search 4: Taxonomy

**Results:** 5 fashion category baselines (luxury apparel, footwear, etc.)

**Time:** ~80ms

**Combined Time:** max(150, 100, 120, 80) = **150ms** (parallel execution)

**Console Output:**
```
📊 Data Sources Found:
  ✓ Brand mentions: 12
  ✓ Enriched lyrics: 28
  ✓ Full lyrics: 15
  ✓ Taxonomy: 5
```

---

## 6. Brand/Item Separation

**File:** `rag-system/app/query_rag.py:51-79`

**Code:**
```python
actual_brands_df, fashion_items_df = separate_brands_and_items(brand_results)
```

**Function Implementation:**
```python
def separate_brands_and_items(brand_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate brand mentions DataFrame into actual brands and fashion items."""

    fashion_items = get_fashion_item_labels()
    # Returns: {'watch', 'bag', 't-shirt', 'boots', 'sneakers', 'jeans', ...}

    # Case-insensitive separation
    is_fashion_item = brand_results_df['brand_name'].str.lower().isin(fashion_items)

    fashion_items_df = brand_results_df[is_fashion_item].copy()
    actual_brands_df = brand_results_df[~is_fashion_item].copy()

    return actual_brands_df, fashion_items_df
```

**Input:**
```python
brand_results_df['brand_name'].unique()
# ['Gucci', 'Versace', 'watch', 'chain', 'Nike']
```

**Classification:**
```python
fashion_items = {'watch', 'chain', ...}

is_fashion_item = ['Gucci', 'Versace', 'watch', 'chain', 'Nike']
    .str.lower()
    .isin(fashion_items)
# [False, False, True, True, False]
```

**Output:**
```python
actual_brands_df['brand_name'].unique()
# ['Gucci', 'Versace', 'Nike']  → Send to Google Trends

fashion_items_df['brand_name'].unique()
# ['watch', 'chain']  → Analyze via taxonomy baselines
```

**Console Output:**
```
🔍 Separated 2 fashion items from 10 brand mentions
```

---

## 7. Trends Analysis

**File:** `rag-system/app/query_rag.py:580-665`

**Code:**
```python
if decision.brand:  # "Gucci"
    print(f"📈 Fetching trends for {decision.brand}...")

    trends_service = TrendsService()
    popularity_analyzer = PopularityAnalyzer(trends_service)

    # Try popularity-based analysis first
    trends_data_list = _analyze_with_popularity_fallback(
        trends_service,
        popularity_analyzer,
        enriched_results,
        actual_brands_df,
        decision.brand
    )

    # If no yearly trends, fall back to aggregate analysis
    if not trends_data_list:
        # Use mention clustering to create analysis window
        mention_dates = trends_service._extract_mention_dates(actual_brands_df)
        # mention_dates = ['2017-01-15', '2017-01-27', '2017-01-27']

        # Find peak cluster
        mention_months = [datetime.fromisoformat(d).replace(day=1) for d in mention_dates]
        # mention_months = [datetime(2017, 1, 1), datetime(2017, 1, 1), datetime(2017, 1, 1)]

        peak_month = Counter(mention_months).most_common(1)[0][0]
        # peak_month = datetime(2017, 1, 1)

        cluster_dates = [d for d in mention_dates if datetime.fromisoformat(d).replace(day=1) == peak_month]
        # cluster_dates = ['2017-01-15', '2017-01-27', '2017-01-27']

        anchor = datetime.fromisoformat(min(cluster_dates))
        # anchor = datetime(2017, 1, 15)

        split_date = anchor.isoformat()  # "2017-01-15"

        # Expand window for baseline + lag
        window_start = (anchor - timedelta(days=30)).strftime('%Y-%m-%d')  # "2016-12-16"
        window_end = (anchor + timedelta(days=60)).strftime('%Y-%m-%d')    # "2017-03-16"

        # Try pre-computed data first
        try:
            trends_data = trends_service.get_brand_trends_from_precomputed(
                brand=decision.brand,
                start_date=window_start,
                end_date=window_end,
                mention_dates=split_date
            )
        except BrandNotFoundError:
            # Fallback to API
            trends_data = trends_service.get_brand_trends(...)
```

### Pre-computed Database Query

**File:** `rag-system/app/services/trends_service.py:200-250`

**SQL Query:**
```sql
SELECT
    label,
    year,
    month,
    period_start,
    trend_mean,
    trend_max,
    trend_min,
    n_points
FROM brand_trends_monthly
WHERE LOWER(label) = LOWER('Gucci')
  AND period_start BETWEEN '2016-12-16' AND '2017-03-16'
ORDER BY year, month
```

**Query Results:**
| label | year | month | period_start | trend_mean | trend_max | trend_min | n_points |
|-------|------|-------|-------------|-----------|----------|----------|----------|
| Gucci | 2016 | 12 | 2016-12-01 | 42.5 | 48 | 38 | 4 |
| Gucci | 2017 | 1 | 2017-01-01 | 58.2 | 67 | 52 | 5 |
| Gucci | 2017 | 2 | 2017-02-01 | 72.8 | 78 | 68 | 4 |
| Gucci | 2017 | 3 | 2017-03-01 | 73.9 | 79 | 70 | 5 |

**Time:** ~10ms

### Metrics Calculation

**Code:**
```python
# Split date: "2017-01-15" (first mention)
split_date = datetime.fromisoformat("2017-01-15")

# Pre-mention: before 2017-01-15
pre_df = df[df['period_start'] < split_date]
# pre_df = [2016-12 row]

pre_avg = pre_df['trend_mean'].mean()
# pre_avg = 42.5

# Post-mention: 2017-01-15 onwards
post_df = df[df['period_start'] >= split_date]
# post_df = [2017-01, 2017-02, 2017-03 rows]

post_avg = post_df['trend_mean'].mean()
# post_avg = (58.2 + 72.8 + 73.9) / 3 = 68.3

# Percent change
percent_change = ((post_avg - pre_avg) / pre_avg) * 100
# percent_change = ((68.3 - 42.5) / 42.5) * 100 = 60.7%
```

**Output:**
```python
trends_data = TrendsResponse(
    brand="Gucci",
    timeframe="2016-12-16 to 2017-03-16",
    pre_mention_avg=42.5,
    post_mention_avg=68.3,
    percent_change=60.7,
    average_interest=61.1,
    data=[
        TrendsDataPoint(date=datetime(2016, 12, 1), value=42.5),
        TrendsDataPoint(date=datetime(2017, 1, 1), value=58.2),
        TrendsDataPoint(date=datetime(2017, 2, 1), value=72.8),
        TrendsDataPoint(date=datetime(2017, 3, 1), value=73.9),
    ]
)
```

**Console Output:**
```
📊 Found 1 trend analyses
  📊 Aggregate analysis: 12 total mentions
     Pre: 42.5, Post: 68.3, Change: +60.7%
```

---

## 8. Context Formatting

**File:** `rag-system/app/services/synthesizer.py:78-218`

**Code:**
```python
synth = FashionSynthesizer()
insight = synth.generate_insight(
    question,
    actual_brands_df,
    enriched_results,
    lyrics_results,
    taxonomy_results,
    trends_data_list,
    decision.comparative_query,
    None,  # category_baseline
    fashion_trends,
    aggregation_data
)
```

**Internal Processing:**
```python
def generate_insight(self, question, brand_mentions_df, ...):
    # Format brand context (top 30)
    brand_context = self._format_brand_context(brand_mentions_df)

    # Format trends data
    trends_str = ""
    for cluster_data in trends_data_list:
        month = cluster_data['month']
        count = cluster_data['mention_count']
        trends = cluster_data['trends']

        trends_str += f"📊 {month} Cluster ({count} mentions):\n"
        trends_str += f"   Brand: {trends.brand}\n"
        trends_str += f"   Window: {trends.timeframe}\n"
        trends_str += f"   Pre-cluster: {trends.pre_mention_avg}\n"
        trends_str += f"   Post-cluster: {trends.post_mention_avg}\n"
        trends_str += f"   Change: {trends.percent_change:+.1f}%\n\n"

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nData:\n{brand_context}{trends_str}"}
    ]

    return messages
```

**Formatted Context (sent to GPT-5):**
```
Question: Did Gucci spike after Migos Culture album in January 2017?

BRAND MENTIONS:
1. Migos (Hip-Hop) - 'T-Shirt' (1/15/2017)
   Brand: Gucci (Luxury) | Context: Artist: Migos
Song: T-Shirt
Brand: Gucci
Context: Gucci my t-shirt, Versace my shoes, riding in a...

2. Migos (Hip-Hop) - 'Bad and Boujee' (1/27/2017)
   Brand: Gucci (Luxury) | Context: Artist: Migos
Song: Bad and Boujee
Brand: Gucci
Context: rain drop, drop top, smoking on cookie in the...

3. Migos (Hip-Hop) - 'Slippery' (1/27/2017)
   Brand: Gucci (Luxury) | Context: Artist: Migos
Song: Slippery
Brand: Gucci
Context: Gucci flip flops, fuck it, hit your bitch in my...

[... 9 more brand mentions ...]


FASHION ITEM MENTIONS (ENRICHED):
• Migos - 'T-Shirt'
  Said: 'Guwop' → Actual item: Gucci
  Context: Mama told me not to sell work...

• Migos - 'Bad and Boujee'
  Said: 'Gucci' → Actual item: Gucci
  Context: rain drop, drop top, smoking on cookie in the hot box...

[... 18 more enriched mentions ...]


FULL LYRICS CONTEXT:
• Migos - 'T-Shirt'
  Mama told me not to sell work / Seventeen five, same color...

[... 9 more full lyrics ...]


CATEGORY BASELINE TRENDS:
• luxury apparel (Luxury)
  Peak: 85, Avg: 52.3, Recent: 48.7


📊 aggregate Cluster (12 mentions):
   Brand: Gucci
   Window: 2016-12-16 to 2017-03-16
   Pre-cluster: 42.5
   Post-cluster: 68.3
   Change: +60.7%
```

---

## 9. LLM Synthesis

**File:** `rag-system/app/services/synthesizer.py:220-224`

**Code:**
```python
return self.llm.create_completion(
    response_model=FashionInsight,
    messages=messages,
    model="gpt-5"
)
```

**API Request:**
```http
POST https://api.openai.com/v1/chat/completions
Content-Type: application/json
Authorization: Bearer sk-...

{
    "model": "gpt-5",
    "messages": [
        {
            "role": "system",
            "content": "You are an expert fashion analytics AI specializing in causal inference..."
        },
        {
            "role": "user",
            "content": "Question: Did Gucci spike after Migos Culture album in January 2017?\n\nBRAND MENTIONS:\n..."
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "FashionInsight",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "data_quality": {"type": "string"}
                },
                "required": ["summary", "key_findings", "data_quality"]
            }
        }
    }
}
```

**API Response (Raw JSON):**
```json
{
    "choices": [{
        "message": {
            "content": "{\"summary\": \"Yes, Gucci experienced a significant +60.7% spike in search interest after the Migos Culture album release in January 2017. The album featured multiple Gucci mentions, with 'Bad and Boujee' becoming a viral #1 hit that drove substantial consumer interest.\", \"key_findings\": [\"• Migos - 'T-Shirt' (1/15/2017): First album mention, pre-release buzz with 'Guwop' slang reference\", \"• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions, became #1 Billboard hit with massive cultural impact\", \"• Pre-album baseline: 42.5 (Dec 2016) → Post-release: 68.3 (Jan-Mar 2017) = +60.7% increase\", \"• Peak trend value of 78 occurred 3 weeks post-release (mid-February 2017)\", \"• Excess impact vs luxury category baseline: +48.2% (Gucci +60.7% vs luxury apparel +12.5%)\"], \"data_quality\": \"sufficient\"}"
        }
    }]
}
```

**instructor Validation:**
```python
# Parse JSON
raw_json = json.loads(response.choices[0].message.content)

# Validate against Pydantic schema
insight = FashionInsight(**raw_json)

# Pydantic checks:
# ✓ summary is str?
# ✓ key_findings is List[str]?
# ✓ data_quality in ['sufficient', 'partial', 'insufficient']?
# ✓ All required fields present?

# If validation passes:
return insight
```

**Validated Output:**
```python
insight = FashionInsight(
    summary="Yes, Gucci experienced a significant +60.7% spike in search interest after the Migos Culture album release in January 2017. The album featured multiple Gucci mentions, with 'Bad and Boujee' becoming a viral #1 hit that drove substantial consumer interest.",

    key_findings=[
        "• Migos - 'T-Shirt' (1/15/2017): First album mention, pre-release buzz with 'Guwop' slang reference",
        "• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions, became #1 Billboard hit with massive cultural impact",
        "• Pre-album baseline: 42.5 (Dec 2016) → Post-release: 68.3 (Jan-Mar 2017) = +60.7% increase",
        "• Peak trend value of 78 occurred 3 weeks post-release (mid-February 2017)",
        "• Excess impact vs luxury category baseline: +48.2% (Gucci +60.7% vs luxury apparel +12.5%)"
    ],

    data_quality="sufficient"
)
```

**Time:** ~2.3 seconds

---

## 10. Output Formatting

**File:** `rag-system/app/query_rag.py:698-730`

**Code:**
```python
# Print summary
print(f"\n{insight.summary}\n")

# Print key findings
for finding in insight.key_findings:
    print(f"{finding}")

# Print data quality
print(f"\nQuality: {insight.data_quality}\n")

# Display brand mentions (debugging/transparency)
if not actual_brands_df.empty:
    unique_artists = actual_brands_df['artist_name'].unique()
    print(f"Artists: {', '.join([str(a) for a in unique_artists if a])}\n")

    recent = actual_brands_df.sort_values('release_date', ascending=False).head(10)
    print("Sample brand mentions:")

    display_cols = ['artist_name', 'song_title', 'release_date']
    if 'popularity_weight' in recent.columns:
        display_cols.append('popularity_weight')
    print(recent[display_cols].to_string(index=False))

print(f"\n{'='*80}\n")
```

**Final Console Output:**
```
================================================================================
Query: Did Gucci spike after Migos Culture album in January 2017?
================================================================================

📅 Date Range: 2017-01-01 to 2017-01-31

📊 Data Sources Found:
  ✓ Brand mentions: 12
  ✓ Enriched lyrics: 28
  ✓ Full lyrics: 15
  ✓ Taxonomy: 5

📈 Fetching trends for Gucci...

📊 Found 1 trend analyses
  📊 Aggregate analysis: 12 total mentions
     Pre: 42.5, Post: 68.3, Change: +60.7%

Yes, Gucci experienced a significant +60.7% spike in search interest after the
Migos Culture album release in January 2017. The album featured multiple Gucci
mentions, with 'Bad and Boujee' becoming a viral #1 hit that drove substantial
consumer interest.

• Migos - 'T-Shirt' (1/15/2017): First album mention, pre-release buzz with 'Guwop' slang reference
• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions, became #1 Billboard hit with massive cultural impact
• Pre-album baseline: 42.5 (Dec 2016) → Post-release: 68.3 (Jan-Mar 2017) = +60.7% increase
• Peak trend value of 78 occurred 3 weeks post-release (mid-February 2017)
• Excess impact vs luxury category baseline: +48.2% (Gucci +60.7% vs luxury apparel +12.5%)

Quality: sufficient

Artists: Migos

Sample brand mentions:
artist_name  song_title      release_date  popularity_weight
Migos        Bad and Boujee  1/27/2017     95.2
Migos        Slippery        1/27/2017     82.1
Migos        T-Shirt         1/15/2017     87.5
Migos        Get Right Witcha 1/27/2017    74.3
Migos        Call Casting    1/27/2017     68.9

================================================================================

Total time: ~3.8 seconds
```

---

## Performance Breakdown

| Phase | Time | % of Total |
|-------|------|-----------|
| 1. Query entry | <1ms | <1% |
| 2. Date extraction | 5ms | <1% |
| 3. Query classification (GPT-5) | 850ms | 22% |
| 4. Embedding generation | 142ms | 4% |
| 5. Parallel vector search | 150ms | 4% |
| 6. Brand/item separation | 10ms | <1% |
| 7. Trends analysis (DB) | 10ms | <1% |
| 8. Context formatting | 50ms | 1% |
| 9. LLM synthesis (GPT-5) | 2,300ms | 61% |
| 10. Output formatting | 50ms | 1% |
| **Total** | **~3,767ms** | **100%** |

**Bottleneck:** LLM API calls (83% of time)

---

## Key Takeaways

1. **Embeddings enable semantic search** - "Gucci" matches "Guwop" (slang)
2. **Parallel execution saves time** - 4 searches in 150ms vs 450ms sequential
3. **Pre-computed data is critical** - 10ms vs 8-9s for API
4. **Structured outputs prevent errors** - Pydantic validates LLM responses
5. **Context formatting is crucial** - Clear prompts → better LLM outputs
6. **Temporal analysis drives insights** - Pre/post comparison shows causality

---

## Next Steps

- See [README.md](README.md) for quick overview
- See [ARCHITECTURE.md](ARCHITECTURE.md) for component details
- Run your own queries with `python app/query_rag.py`
