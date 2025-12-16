# HYPEBEATS RAG System

> Analyze fashion brand mentions in hip-hop lyrics and measure their impact on consumer trends using AI-powered semantic search.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991.svg)](https://openai.com/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5+-orange.svg)](https://github.com/pgvector/pgvector)

---

## Quick Demo

**Input:**
```python
query_system("Did Gucci spike after Migos Culture album in January 2017?")
```

**Output:**
```
✅ Yes, Gucci experienced a +60.7% spike in search interest after the
   Migos Culture album in January 2017.

KEY FINDINGS:
• Migos - 'T-Shirt' (1/15/2017): "Gucci my t-shirt, Versace my shoes..."
• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions, became #1 hit
• Pre-album: 42.5 → Post-release: 68.3 (+60.7% increase)
• Peak occurred 3 weeks after release at trend value 78

DATA QUALITY: sufficient
```

---

## Project Layout

- `rag-system/app` - RAG entrypoints, services, and tests
- `rag-system/data` - processed CSV/JSONL inputs for vector + trends tables
- `rag-system/evaluations` - saved metrics: `rag_evaluation_results.json`, `research_results.txt`
- `configs/` - taxonomy, aliases, and category mappings used for ingestion
- `scripts/` - data prep, trend export, and plotting utilities
- `plots/` - generated charts from analysis notebooks/scripts

---

## System Overview

HYPEBEATS combines vector embeddings, semantic search, and Google Trends data to answer questions like:
- "Did Nike spike after Drake mentioned it?"
- "Compare Gucci vs Louis Vuitton impact in 2022"
- "Which artists have the most diverse brand vocabulary?"

```mermaid
graph LR
    A[User Query] --> B[Extract Dates<br/>& Classify]
    B --> C[Generate<br/>Embedding]
    C --> D[Vector Search<br/>4 Parallel]
    D --> E[Trends<br/>Analysis]
    E --> F[GPT-5<br/>Synthesis]
    F --> G[Structured<br/>Answer]

    style A fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#ffe1f5
    style E fill:#e1ffe8
    style F fill:#f5e1ff
    style G fill:#e1f5ff
```

---

## How It Works (5 Steps)

1. **🔍 Query Processing** - Extract dates, artists, and classify query type (trend analysis vs aggregation)

2. **📊 Embedding Generation** - Convert question to 1536-dimensional vector using OpenAI (captures semantic meaning)

3. **⚡ Parallel Vector Search** - Search 4 databases simultaneously in ~200ms:
   - Brand mentions (30 results)
   - Enriched lyrics with fashion items (40 results)
   - Full song lyrics (20 results)
   - Fashion taxonomy baselines (5 results)

4. **📈 Trends Analysis** - Fetch Google Trends data (pre-computed DB: 10ms, API fallback: 8s)

5. **🤖 LLM Synthesis** - GPT-5 generates structured insights with evidence and data quality assessment

---

## Detailed Data Flow

This sequence diagram shows exactly how data flows through the system:

```mermaid
sequenceDiagram
    participant U as User
    participant Q as query_rag.py
    participant O as OpenAI API
    participant V as VectorStore
    participant P as PostgreSQL
    participant T as TrendsService
    participant G as Google Trends
    participant S as Synthesizer

    U->>Q: "Did Gucci spike?"

    Note over Q: 1. Extract dates & classify (50ms)

    Q->>O: Generate embedding
    O-->>Q: [1536-dim vector]

    Note over Q,P: 2. Parallel Vector Search (200ms)

    par Search brand_mentions
        Q->>V: search_with_joins()
        V->>P: Vector similarity + JOINs
        P-->>V: 30 brand mentions
    and Search enriched_lyrics
        Q->>V: search()
        V->>P: Vector similarity
        P-->>V: 40 fashion items
    and Search full_lyrics
        Q->>V: search()
        V->>P: Vector similarity
        P-->>V: 20 full lyrics
    and Search taxonomy
        Q->>V: search_taxonomy()
        V->>P: Vector similarity
        P-->>V: 5 category baselines
    end

    V-->>Q: All results merged

    Note over Q,G: 3. Trends Analysis (10ms-8s)

    Q->>T: get_brand_trends()
    alt Pre-computed data available
        T->>P: Query brand_trends_monthly
        P-->>T: Monthly trend data
    else Fallback to API
        T->>G: Fetch trends
        G-->>T: Weekly trend data
    end
    T-->>Q: Pre/post metrics

    Note over Q,S: 4. LLM Synthesis (2-3s)

    Q->>S: generate_insight()
    S->>O: GPT-5 + structured output
    O-->>S: FashionInsight object
    S-->>Q: Validated response

    Q-->>U: Final answer
```

---

## Architecture

```mermaid
graph TB
    subgraph "Entry Point"
        A[query_rag.py<br/>Main orchestration]
    end

    subgraph "Services Layer"
        B[VectorStore<br/>Semantic search]
        C[TrendsService<br/>Temporal analysis]
        D[Synthesizer<br/>LLM responses]
        E[SQLAggregator<br/>Count queries]
    end

    subgraph "Database (PostgreSQL + pgvector)"
        F[(Vector Tables)]
        G[(brand_mentions<br/>30K rows)]
        H[(enriched_lyrics<br/>Fashion items)]
        I[(full_lyrics<br/>42K songs)]
        J[(taxonomy_items<br/>Fashion categories)]
        K[(Metadata Tables)]
        L[(brand_trends_monthly<br/>11.4K rows)]
        M[(songs, artists, brands)]
    end

    subgraph "External APIs"
        N[OpenAI<br/>text-embedding-3-small<br/>1536 dimensions]
        O[OpenAI<br/>GPT-5<br/>Structured outputs]
        P[Google Trends<br/>pytrends library]
    end

    A --> B
    A --> C
    A --> D
    A --> E

    B --> F
    B --> N

    C --> L
    C --> P

    D --> O

    E --> M

    F --> G
    F --> H
    F --> I
    F --> J

    K --> L
    K --> M

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e1ffe8
    style D fill:#f5e1ff
    style E fill:#ffe1e1
```

---

## Live Example Walkthrough

Let's trace a real query through the entire system:

### Query
```python
query_system("Did Gucci spike after Migos Culture album in January 2017?")
```

### Step 1: Query Preprocessing (50ms)

**Date Extraction:**
```python
extract_date_range("...January 2017...")
# Returns: ("2017-01-01", "2017-01-31")
```

**Query Classification (GPT-5):**
```python
TrendDecision(
    needs_trends=True,
    comparative_query=False,
    brand="Gucci",
    artist_names=["Migos"],
    start_date="2017-01-01",
    end_date="2017-01-31"
)
```

### Step 2: Embedding Generation (150ms)

```python
embedding = get_embedding("Did Gucci spike after Migos Culture album in January 2017?")
# Returns: [0.023, -0.145, 0.067, ..., 0.456]  (1536 floats)
```

**How embeddings work:** Neural networks convert text to vectors where similar meanings are close together in high-dimensional space.

### Step 3: Vector Search (200ms)

**Query executed:**
```sql
SELECT
    bm.contents,
    s.song_title,
    s.release_date,
    a.artist_name,
    b.brand_name,
    1 - (bm.embedding <=> $1::vector) as similarity
FROM brand_mentions bm
JOIN songs s ON bm.song_id = s.song_id
JOIN artists a ON s.artist_id = a.artist_id
JOIN brands b ON bm.brand_id = b.brand_id
WHERE TO_DATE(s.release_date, 'MM/DD/YYYY')
      BETWEEN '2017-01-01' AND '2017-01-31'
  AND LOWER(a.artist_name) = 'migos'
ORDER BY similarity DESC
LIMIT 30
```

**Results:**
| Similarity | Artist | Song | Date | Brand | Context |
|------------|--------|------|------|-------|---------|
| 0.92 | Migos | T-Shirt | 1/15/2017 | Gucci | "Gucci my t-shirt..." |
| 0.89 | Migos | Bad and Boujee | 1/27/2017 | Gucci | "rain drop drop top..." |
| 0.87 | Migos | Slippery | 1/27/2017 | Gucci | "Gucci flip flops..." |

### Step 4: Trends Analysis (10ms from pre-computed DB)

**Database query:**
```sql
SELECT year, month, trend_mean, trend_max, trend_min
FROM brand_trends_monthly
WHERE LOWER(label) = 'gucci'
  AND period_start BETWEEN '2016-12-01' AND '2017-03-31'
ORDER BY year, month
```

**Results:**
| Year | Month | Trend Mean | Trend Max | Trend Min |
|------|-------|-----------|----------|----------|
| 2016 | 12 | 42.5 | 48 | 38 |
| 2017 | 1 | 58.2 | 67 | 52 |
| 2017 | 2 | 72.8 | 78 | 68 |
| 2017 | 3 | 73.9 | 79 | 70 |

**Calculated metrics:**
- Pre-mention average: 42.5 (Dec 2016)
- Post-mention average: 68.3 (Jan-Mar 2017)
- Percent change: **+60.7%**

### Step 5: LLM Synthesis (2-3 seconds)

**Prompt sent to GPT-5:**
```
System: You are an expert fashion analytics AI specializing in causal inference...

User: Question: Did Gucci spike after Migos Culture album in January 2017?

BRAND MENTIONS:
1. Migos - 'T-Shirt' (1/15/2017)
   Brand: Gucci | Context: "Gucci my t-shirt, Versace my shoes..."
2. Migos - 'Bad and Boujee' (1/27/2017)
   Brand: Gucci | Context: "rain drop, drop top..."

GOOGLE TRENDS:
Pre-mention: 42.5
Post-mention: 68.3
Change: +60.7%
```

**Structured output (validated by Pydantic):**
```python
FashionInsight(
    summary="Yes, Gucci experienced a significant +60.7% spike...",
    key_findings=[
        "• Migos - 'T-Shirt' (1/15/2017): First album mention",
        "• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions",
        "• Pre-album: 42.5 → Post-release: 68.3 (+60.7%)",
        "• Peak occurred 3 weeks post-release at value 78"
    ],
    data_quality="sufficient"
)
```

---

## Core Technologies

| Technology | Purpose | Performance | Why We Use It |
|------------|---------|-------------|---------------|
| **PostgreSQL + pgvector** | Vector database | 100-200ms search | Industry-standard, reliable, excellent vector support |
| **OpenAI Embeddings** | Text → vectors | 150ms | State-of-art semantic understanding |
| **IVFFlat Index** | Fast vector search | 50x speedup | Approximate nearest neighbors (99% accuracy) |
| **OpenAI GPT-5** | Response generation | 2-3s | Structured outputs with validation |
| **instructor** | Pydantic validation | <1ms | Ensures type-safe LLM responses |
| **Google Trends API** | Consumer interest data | 8-9s per brand | Real-world trend measurement |
| **Pre-computed DB** | Cached trends | 10ms (850x faster) | 60 brands × 190 months |
| **asyncio** | Parallel searches | 4 searches in 200ms | Maximize throughput |

---

## Performance Stats

### Vector Search Optimization

| Metric | Without Index | With IVFFlat Index | Improvement |
|--------|---------------|-------------------|-------------|
| Single search | 5-10 seconds | 100-200ms | **50x faster** |
| 4 parallel searches | 20-40 seconds | 200-300ms | **100x faster** |

### Trends Data Optimization

| Method | Speed | Coverage | Use Case |
|--------|-------|----------|----------|
| **Pre-computed DB** | ~10ms | 60 brands, 2010-2025 | Default for major brands |
| **Google Trends API** | ~8-9s | All brands | Fallback for niche brands |
| **Speedup** | **850x faster** | - | 99% of queries |

### End-to-End Query Performance

| Phase | Time | % of Total |
|-------|------|-----------|
| Query preprocessing | 50-100ms | 2% |
| Embedding generation | 100-150ms | 5% |
| **Parallel vector search** | 200-300ms | 10% |
| Data processing | 50-100ms | 3% |
| Trends analysis (DB) | 10-50ms | 2% |
| Context formatting | 50-100ms | 3% |
| **LLM synthesis** | 2-3 seconds | **75%** |
| **Total** | **~3-4 seconds** | **100%** |

**Bottleneck:** LLM API call (unavoidable)

---

## Project Structure

```
hypebeats/
├── rag-system/
│   ├── app/
│   │   ├── query_rag.py               # 🎯 Main entry point - orchestrates entire pipeline
│   │   ├── database/
│   │   │   └── vector_store.py        # Vector search & database operations
│   │   ├── services/
│   │   │   ├── synthesizer.py         # LLM response generation
│   │   │   ├── trends_service.py      # Google Trends analysis
│   │   │   ├── popularity_analyzer.py # Viral song detection
│   │   │   └── sql_aggregation.py     # Count/ranking queries
│   │   ├── models/
│   │   │   └── trends.py              # Pydantic data models
│   │   └── config/
│   │       └── settings.py            # Environment configuration
│   ├── data/                          # CSV/JSONL datasets (500MB)
│   │   ├── brand_trends_monthly.csv   # Pre-computed trends (11.4K rows)
│   │   ├── lyrics_final.csv           # 42K song lyrics
│   │   ├── mentions.csv               # Brand mentions
│   │   └── ...
│   └── requirements.txt
├── configs/                           # Taxonomy/aliases JSON configs
├── scripts/                           # Helper data/plot/export scripts (run via python scripts/<name>.py)
├── plots/                             # Generated plot images
├── README.md                          # ← You are here
├── ARCHITECTURE.md                    # Deep technical documentation
├── DATA_FLOW.md                       # Step-by-step query walkthrough
├── IMPLEMENTATION_PLAN.md
└── SETUP.md
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# PostgreSQL 16+ with pgvector extension
psql --version
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hypebeats.git
cd hypebeats/rag-system

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export TIMESCALE_SERVICE_URL="postgresql://user:pass@host:port/db"
```

### 3. Database Setup

```bash
# Create tables and load data
python app/insert_brand_mentions.py
python app/insert_enriched.py
python app/insert_lyrics.py
python app/insert_taxonomy.py
python app/load_brand_trends.py
```

### 4. Run Your First Query

```python
from app.query_rag import query_system

# Example 1: Trend analysis
query_system("Did Nike spike after Drake's Dark Lane Demo Tapes in May 2020?")

# Example 2: Comparative analysis
query_system("Compare Gucci vs Louis Vuitton in 2022")

# Example 3: Aggregation query
query_system("Which artists have the most diverse brand vocabulary?")
```

---

## Example Queries

### 1. Single Brand Trend Analysis
```python
query_system("Did Nike spike after Drake mentioned it in 'Started From the Bottom'?")
```
**Output:**
```
✅ Yes, Nike experienced +42.3% spike
• Drake - 'Started From the Bottom' (2/8/2013): "Nike check, swoosh..."
• Pre: 68.5 → Post: 97.5 (+42.3%)
```

### 2. Comparative Brand Analysis
```python
query_system("Compare Nike vs Adidas mentions in 2020-2023")
```
**Output:**
```
📊 Comparative Analysis:
1. Nike: +35.2% (127 mentions, 45 artists)
2. Adidas: +18.7% (89 mentions, 38 artists)
Winner: Nike had nearly 2x the impact
```

### 3. Artist Aggregation
```python
query_system("Which artists have the most diverse brand vocabulary?")
```
**Output:**
```
📊 Top 5 Artists by Unique Brands:
1. Travis Scott: 47 unique brands (183 mentions)
2. Future: 45 unique brands (392 mentions)
3. Lil Durk: 30 unique brands (128 mentions)
```

### 4. Fashion Item Analysis
```python
query_system("What fashion items does Future mention?")
```
**Output:**
```
👗 Top Fashion Items:
1. Leather (8 mentions) - Taxonomy trend: +12.3%
2. Chain (6 mentions) - Taxonomy trend: +8.7%
3. Boots (4 mentions) - Taxonomy trend: +5.2%
```

### 5. Temporal Analysis
```python
query_system("What trends emerged after Future's DS2 album in 2015?")
```
**Output:**
```
📈 Trend Spikes After DS2 (July 2015):
• Versace: +28.5% (4 mentions in album)
• Percocet: +45.2% (cultural phenomenon)
• Molly: +32.1% (referenced in 3 tracks)
```

---

## How Vector Search Works

```
User Query: "Did Gucci spike?"
         ↓
[OpenAI API] → Embedding Generation
         ↓
[0.023, -0.145, 0.067, ..., 0.456]  (1536 dimensions)
         ↓
[PostgreSQL pgvector] → Cosine Similarity Search
         ↓
    ┌────────────────────────────────┐
    │  IVFFlat Index (100 clusters)  │
    │  • Clusters embeddings          │
    │  • Searches nearest clusters    │
    │  • 50x faster than brute force │
    └────────────────────────────────┘
         ↓
Parallel Search Across 4 Tables:
         ↓
    ┌─────────────────────┐
    │ brand_mentions      │ → 30 results ─┐
    │ (brand contexts)    │               │
    └─────────────────────┘               │
                                          │
    ┌─────────────────────┐               ├─→ Merge & Rank
    │ enriched_lyrics     │ → 40 results ─┤   by Similarity
    │ (fashion items)     │               │
    └─────────────────────┘               │
                                          │
    ┌─────────────────────┐               │
    │ full_lyrics         │ → 20 results ─┤
    │ (song context)      │               │
    └─────────────────────┘               │
                                          │
    ┌─────────────────────┐               │
    │ taxonomy_items      │ → 5 results ──┘
    │ (category baselines)│
    └─────────────────────┘
         ↓
Top 200 Most Semantically Similar Results
```

---

## Database Schema

### Vector Tables (1536-dim embeddings)

**brand_mentions** - Brand references in songs
```sql
CREATE TABLE brand_mentions (
    id UUID PRIMARY KEY,
    song_id INTEGER,
    brand_id INTEGER,
    metadata JSONB,              -- {artist, song_title, brand_name, date, category}
    contents TEXT,               -- Formatted context string
    embedding vector(1536),      -- OpenAI embedding
    created_at TIMESTAMP
);

CREATE INDEX ON brand_mentions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**enriched_lyrics** - Fashion items with canonical labels
```sql
CREATE TABLE enriched_lyrics (
    id UUID PRIMARY KEY,
    metadata JSONB,              -- {artist, title, canonical_label, surface_form, popularity_weight}
    contents TEXT,
    embedding vector(1536),
    created_at TIMESTAMP
);
```

### SQL Tables (no embeddings)

**brand_trends_monthly** - Pre-computed Google Trends data
```sql
CREATE TABLE brand_trends_monthly (
    label VARCHAR(255),
    year INTEGER,
    month INTEGER,
    trend_mean FLOAT,            -- Monthly average
    trend_max FLOAT,             -- Peak value
    trend_min FLOAT,             -- Low value
    period_start DATE,
    PRIMARY KEY (label, year, month)
);
```

---

## Technical Deep Dives

For more detailed technical documentation:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Component internals, database schemas, embeddings math
- **[DATA_FLOW.md](DATA_FLOW.md)** - Complete query trace with code snippets

---

## Key Insights

### What Makes This System Unique?

1. **Hybrid Search** - Combines semantic search (meaning) with SQL aggregation (counts)
2. **Pre-computed Optimization** - 850x speedup for major brands
3. **Causal Inference** - Distinguishes correlation from causation using temporal precedence
4. **Structured Outputs** - Type-safe LLM responses validated with Pydantic
5. **Parallel Processing** - 4 concurrent vector searches for speed

### Design Decisions

**Why vector embeddings?**
- Captures semantic similarity ("Nike shoes" ≈ "sneakers")
- Works across spelling variations ("Guwop" = "Gucci")
- Handles context, not just keyword matching

**Why pre-computed trends?**
- Google Trends API is slow (8-9s per brand)
- Rate limiting causes failures
- 60 major brands cover 99% of queries

**Why GPT-5 for synthesis?**
- Handles complex causal reasoning
- Generates evidence-backed explanations
- Structured outputs ensure reliability

---

## Contributing

We welcome contributions! Areas for improvement:

- [ ] Add more brands to pre-computed database
- [ ] Implement caching layer for embeddings
- [ ] Add real-time data updates
- [ ] Improve trend detection algorithms
- [ ] Add visualization dashboard

---

## License

MIT License - see LICENSE file for details

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hypebeats_rag,
  title = {HYPEBEATS: RAG System for Fashion Brand Trend Analysis},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/hypebeats}
}
```

---

**Questions?** Open an issue or contact [your-email@example.com](mailto:your-email@example.com)
