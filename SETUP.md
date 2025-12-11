# HYPEBEATS RAG System - Setup Guide

> Step-by-step guide to get the system running from scratch

**Estimated Time:** 15-20 minutes

---

## Prerequisites

Before starting, ensure you have the following installed:

- [ ] **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop/)
- [ ] **Python 3.11+** - Check with `python --version`
- [ ] **TablePlus** (optional but recommended) - [Download](https://tableplus.com/)
- [ ] **OpenAI API Key** - Get from [platform.openai.com](https://platform.openai.com/)

---

## Step 1: Clone & Navigate to Project

```bash
# Navigate to project directory
cd /Users/aaditya/Desktop/HYPEBEATS_GH/Untitled/hypebeats/rag-system
```

**Verify you're in the right place:**
```bash
ls
# Should see: app/ data/ docker/ requirements.txt .env
```

---

## Step 2: Start Docker PostgreSQL Container

### Option A: Using Docker Compose (Recommended)

```bash
# Navigate to docker directory
cd docker

# Start PostgreSQL with pgvector
docker-compose up -d
```

**Expected Output:**
```
[+] Running 2/2
 ✔ Network docker_default          Created
 ✔ Container hypebeats-postgres     Started
```

**Verify container is running:**
```bash
docker ps
```

**Expected:**
```
CONTAINER ID   IMAGE                    STATUS         PORTS                    NAMES
abc123def456   ankane/pgvector:latest   Up 10 seconds  0.0.0.0:5432->5432/tcp   hypebeats-postgres
```

### Option B: Using Docker Run (Alternative)

```bash
docker run -d \
  --name hypebeats-postgres \
  -e POSTGRES_DB=hypebeats_rag \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest
```

---

## Step 3: Verify Database is Ready

**Test connection:**
```bash
# Wait 5-10 seconds for database to initialize, then:
docker exec -it hypebeats-postgres psql -U postgres -d hypebeats_rag -c "SELECT version();"
```

**Expected Output:**
```
                                                 version
---------------------------------------------------------------------------------------------------------
 PostgreSQL 16.x on x86_64-pc-linux-gnu, compiled by gcc...
(1 row)
```

**Check pgvector extension:**
```bash
docker exec -it hypebeats-postgres psql -U postgres -d hypebeats_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Expected:**
```
CREATE EXTENSION
```

---

## Step 4: Connect with TablePlus

### Launch TablePlus

1. Open TablePlus
2. Click **Create a new connection**
3. Select **PostgreSQL**

### Connection Settings

```
Name:       HYPEBEATS RAG
Host:       localhost
Port:       5432
User:       postgres
Password:   password
Database:   hypebeats_rag
```

**Connection String (alternative):**
```
postgresql://postgres:password@localhost:5432/hypebeats_rag
```

### Test Connection

Click **Test** button (bottom left) → Should show **✓ Connection is OK**

Click **Connect** → You should now see the database (currently empty)

---

## Step 5: Set Up Python Environment

### Create Virtual Environment (Recommended)

```bash
# Navigate back to rag-system directory
cd /Users/aaditya/Desktop/HYPEBEATS_GH/Untitled/hypebeats/rag-system

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

**Verify activation:**
```bash
which python
# Should show: /Users/aaditya/.../hypebeats/rag-system/.venv/bin/python
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Output:**
```
Collecting openai>=1.12.0
Collecting psycopg[binary]
Collecting pgvector
Collecting pandas
...
Successfully installed openai-1.x.x psycopg-3.x.x ...
```

**Verify installation:**
```bash
pip list | grep -E "(openai|psycopg|pgvector|instructor)"
```

**Expected:**
```
instructor        1.x.x
openai            1.x.x
pgvector          0.x.x
psycopg           3.x.x
```

---

## Step 6: Configure Environment Variables

### Check .env File

```bash
cat .env
```

**Should contain:**
```
OPENAI_API_KEY=sk-proj-...
TIMESCALE_SERVICE_URL=postgresql://postgres:password@localhost:5432/hypebeats_rag
```

### Update API Key (if needed)

If the API key is invalid or you have your own:

```bash
# Open .env in your editor
nano .env
# OR
code .env

# Replace with your OpenAI API key:
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
TIMESCALE_SERVICE_URL=postgresql://postgres:password@localhost:5432/hypebeats_rag
```

**Test OpenAI connection:**
```bash
python -c "
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Test API call
response = client.embeddings.create(
    input=['test'],
    model='text-embedding-3-small'
)
print(f'✅ OpenAI API working! Embedding dimensions: {len(response.data[0].embedding)}')
"
```

**Expected:**
```
✅ OpenAI API working! Embedding dimensions: 1536
```

---

## Step 7: Load Data into Database

**Run insert scripts in this order:**

### 7.1 Load Brand Trends (Pre-computed Google Trends data)

```bash
cd app
python load_brand_trends.py
```

**Expected Output:**
```
Loading brand trends from data/brand_trends_monthly.csv...
Loaded 11400 rows
✓ Created brand_trends_monthly table
✓ Inserted 11400 trend records
✓ Created indexes
Done! Loaded 60 brands × 190 months
```

**Time:** ~5 seconds

### 7.2 Insert Taxonomy Items

```bash
python insert_taxonomy.py
```

**Expected Output:**
```
Loading taxonomy from data/taxonomy_trends_detailed.csv...
Generating embeddings for 60 taxonomy items...
Embedding generated in 0.143s
...
✓ Created taxonomy_items table
✓ Inserted 60 taxonomy items with embeddings
Done!
```

**Time:** ~10 seconds (60 OpenAI API calls)

### 7.3 Insert Brand Mentions

```bash
python insert_brand_mentions.py
```

**Expected Output:**
```
Loading mentions from data/mentions.csv...
Joining with songs, artists, brands tables...
Generating embeddings for 30000 brand mentions...
Progress: [=========>] 1000/30000 (3.3%)
...
✓ Created brand_mentions table
✓ Created IVFFlat vector index
✓ Inserted 30000 brand mentions
Done!
```

**Time:** ~5 minutes (30K OpenAI API calls, batched)
**Cost:** ~$0.60 (at $0.02 per 1M tokens)

### 7.4 Insert Enriched Lyrics

```bash
python insert_enriched.py
```

**Expected Output:**
```
Loading enriched lyrics from data/lyrics_mentions_enriched_v2.jsonl...
Generating embeddings for 40000 enriched lyrics...
✓ Created enriched_lyrics table
✓ Inserted 40000 enriched lyrics
Done!
```

**Time:** ~6 minutes
**Cost:** ~$0.80

### 7.5 Insert Full Lyrics

```bash
python insert_lyrics.py
```

**Expected Output:**
```
Loading lyrics from data/lyrics_final.csv...
Truncating to 5000 chars per song...
Generating embeddings for 42000 songs...
✓ Created full_lyrics table
✓ Inserted 42000 full lyrics
Done!
```

**Time:** ~7 minutes
**Cost:** ~$3.12

**Total Data Loading Time:** ~20 minutes
**Total Cost:** ~$4.52

---

## Step 8: Verify Data in TablePlus

### Refresh TablePlus

Click the **Refresh** button (top toolbar)

### Check Tables

You should now see these tables:

```
Tables:
├── brand_mentions          (30,000 rows)
├── enriched_lyrics         (40,000 rows)
├── full_lyrics             (42,000 rows)
├── taxonomy_items          (60 rows)
├── brand_trends_monthly    (11,400 rows)
├── songs                   (metadata)
├── artists                 (metadata)
└── brands                  (metadata)
```

### Verify Row Counts

**Run SQL query in TablePlus:**
```sql
SELECT
    'brand_mentions' as table_name,
    COUNT(*) as row_count
FROM brand_mentions
UNION ALL
SELECT 'enriched_lyrics', COUNT(*) FROM enriched_lyrics
UNION ALL
SELECT 'full_lyrics', COUNT(*) FROM full_lyrics
UNION ALL
SELECT 'taxonomy_items', COUNT(*) FROM taxonomy_items
UNION ALL
SELECT 'brand_trends_monthly', COUNT(*) FROM brand_trends_monthly;
```

**Expected Results:**
| table_name | row_count |
|------------|-----------|
| brand_mentions | 30000 |
| enriched_lyrics | 40000 |
| full_lyrics | 42000 |
| taxonomy_items | 60 |
| brand_trends_monthly | 11400 |

### Test Vector Search

**Run this query to test pgvector:**
```sql
SELECT
    metadata->>'song_title' as song,
    metadata->>'artist_name' as artist,
    metadata->>'brand_name' as brand,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
FROM brand_mentions
ORDER BY similarity DESC
LIMIT 5;
```

(Note: You'll need a real embedding vector for actual results)

---

## Step 9: Run Your First Query

### Navigate to App Directory

```bash
cd /Users/aaditya/Desktop/HYPEBEATS_GH/Untitled/hypebeats/rag-system/app
```

### Test Query 1: Single Brand Trend

```bash
python query_rag.py
```

**Default query in script:**
```python
query_system("Did Gucci spike after Migos Culture album in January 2017?")
```

**Expected Output:**
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

Yes, Gucci experienced a significant +60.7% spike in search interest after
the Migos Culture album release in January 2017.

• Migos - 'T-Shirt' (1/15/2017): First album mention
• Migos - 'Bad and Boujee' (1/27/2017): 3 Gucci mentions, #1 hit
• Pre-album: 42.5 → Post-release: 68.3 (+60.7%)

Quality: sufficient

================================================================================
```

**Time:** ~3-4 seconds

### Test Query 2: Comparative Analysis

Edit `query_rag.py` and change the query:

```python
query_system("Compare Nike vs Adidas mentions in 2020")
```

**Run:**
```bash
python query_rag.py
```

### Test Query 3: Aggregation

```python
query_system("Which artists have the most diverse brand vocabulary?")
```

---

## Step 10: Interactive Usage

### Python REPL

```bash
python
```

```python
from query_rag import query_system

# Ask any question
query_system("Did Nike spike after Drake mentioned it?")

# Comparative query
query_system("Compare Gucci vs Louis Vuitton in 2022")

# Aggregation query
query_system("Which songs contain the highest number of brand references?")

# Fashion items
query_system("What fashion items does Future mention?")
```

---

## Troubleshooting

### Issue 1: Docker Container Won't Start

**Error:** `port 5432 is already allocated`

**Solution:**
```bash
# Check if PostgreSQL is already running
lsof -i :5432

# Stop existing PostgreSQL
brew services stop postgresql
# OR
sudo systemctl stop postgresql

# Restart Docker container
docker-compose up -d
```

### Issue 2: OpenAI API Key Invalid

**Error:** `AuthenticationError: Incorrect API key`

**Solution:**
```bash
# Get new API key from platform.openai.com
# Update .env file
echo "OPENAI_API_KEY=sk-proj-NEW_KEY_HERE" > .env
echo "TIMESCALE_SERVICE_URL=postgresql://postgres:password@localhost:5432/hypebeats_rag" >> .env
```

### Issue 3: Database Connection Failed

**Error:** `could not connect to server`

**Check Docker status:**
```bash
docker ps
# If container not running:
docker-compose up -d

# Check logs:
docker logs hypebeats-postgres
```

**Test connection:**
```bash
psql postgresql://postgres:password@localhost:5432/hypebeats_rag
```

### Issue 4: Embedding Generation Too Slow

**Symptoms:** Insert scripts taking >30 minutes

**Causes:**
- Rate limiting from OpenAI
- Network latency
- Large batch size

**Solutions:**
```bash
# Check OpenAI rate limits
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Reduce batch size in insert scripts (edit files):
# Change: batch_size = 100
# To:     batch_size = 10
```

### Issue 5: Out of Memory

**Error:** `MemoryError` during data loading

**Solution:**
```bash
# Process data in smaller chunks
# Edit insert scripts to process in batches of 1000 instead of all at once

# OR increase Docker memory limit:
# Docker Desktop → Settings → Resources → Memory → 8GB
```

### Issue 6: pgvector Extension Not Found

**Error:** `extension "vector" is not available`

**Solution:**
```bash
# Use correct Docker image
docker pull ankane/pgvector:latest

# Verify image
docker images | grep pgvector

# Recreate container
docker-compose down
docker-compose up -d
```

### Issue 7: TablePlus Can't Connect

**Error:** `Connection refused`

**Checklist:**
- [ ] Docker container running: `docker ps`
- [ ] Port 5432 available: `lsof -i :5432`
- [ ] Correct password: `password`
- [ ] Database created: `hypebeats_rag`

**Test with psql:**
```bash
docker exec -it hypebeats-postgres psql -U postgres -d hypebeats_rag
```

---

## Quick Reference

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...
TIMESCALE_SERVICE_URL=postgresql://postgres:password@localhost:5432/hypebeats_rag
```

### Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker logs hypebeats-postgres

# Restart
docker-compose restart

# Remove (WARNING: deletes data)
docker-compose down -v
```

### Database Commands

```bash
# Connect with psql
docker exec -it hypebeats-postgres psql -U postgres -d hypebeats_rag

# Backup database
docker exec hypebeats-postgres pg_dump -U postgres hypebeats_rag > backup.sql

# Restore database
docker exec -i hypebeats-postgres psql -U postgres -d hypebeats_rag < backup.sql

# Check table sizes
docker exec -it hypebeats-postgres psql -U postgres -d hypebeats_rag -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

### Python Commands

```bash
# Activate venv
source .venv/bin/activate

# Deactivate
deactivate

# Install package
pip install package_name

# Update requirements
pip freeze > requirements.txt
```

---

## Data Loading Summary

| Script | Rows | Time | Cost | Purpose |
|--------|------|------|------|---------|
| load_brand_trends.py | 11.4K | 5s | $0 | Pre-computed Google Trends |
| insert_taxonomy.py | 60 | 10s | $0.01 | Fashion category baselines |
| insert_brand_mentions.py | 30K | 5 min | $0.60 | Brand references in songs |
| insert_enriched.py | 40K | 6 min | $0.80 | Fashion items with labels |
| insert_lyrics.py | 42K | 7 min | $3.12 | Full song lyrics |
| **Total** | **~123K** | **~20 min** | **~$4.53** | |

---

## Next Steps

Now that your system is running:

1. **Explore Example Queries** - See [README.md](README.md#example-queries)
2. **Understand Architecture** - Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Learn Data Flow** - Review [DATA_FLOW.md](DATA_FLOW.md)
4. **Customize Queries** - Edit `query_rag.py` for your use cases
5. **Add More Data** - Load additional songs/brands/trends

---

## Success Checklist

- [ ] Docker container running
- [ ] TablePlus connected to database
- [ ] Python environment activated
- [ ] Dependencies installed
- [ ] Environment variables configured
- [ ] All data loaded (5 scripts completed)
- [ ] First query successful
- [ ] Can query via Python REPL

**All done?** You're ready to analyze fashion trends! 🎉

---

## Support

- **Issues:** Open an issue on GitHub
- **Questions:** Check [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- **Contributions:** See CONTRIBUTING.md (if exists)

---

**Estimated Total Setup Time:** 30-40 minutes (including data loading)
