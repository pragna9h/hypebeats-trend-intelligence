# app/query_rag.py
import re
import calendar
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from collections import Counter
from app.database.vector_store import VectorStore
from app.services.synthesizer import FashionSynthesizer
from app.services.trends_service import TrendsService, TrendDecision, BrandNotFoundError, PRECOMPUTED_START_YEAR, PRECOMPUTED_END_YEAR
from app.services.sql_aggregation import SQLAggregator, deduplicate_brand_mentions, route_aggregation_query
from app.services.popularity_analyzer import PopularityAnalyzer
from app.models.trends import TrendsRequest
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FashionTerms(BaseModel):
    """Extracted fashion terms from lyrics."""
    items: list[str] = Field(description="Fashion items mentioned: leather, denim, boots, etc.")

def get_fashion_item_labels() -> set[str]:
    """Return set of known fashion item labels to distinguish from brands.

    Synchronized with LABEL_MAP in trends_service.py to maintain consistency.
    Fashion items are analyzed via taxonomy table, not Google Trends API.
    """
    return {
        # Watches (tracked as brands in Google Trends)
        'watch', 'ap', 'patek', 'rolex', 'patek philippe', 'audemars piguet', 'cartier',
        # Bags/accessories
        'bag', 'tote', 'tote bag', 'crossbody bag',
        # Tops
        't-shirt', 'tee', 'shirt', 'graphic tee', 'flannel shirt',
        # Footwear
        'heels', 'stilettos', 'platform shoes', 'boots', 'combat boots',
        'sneakers', 'slides',
        # Bottoms
        'jeans', 'mom jeans', 'pants', 'shorts',
        # Other apparel
        'cap', 'baseball cap', 'sweater', 'knit sweater',
        'jacket', 'fleece jacket', 'dress', 'skirt',
        # Jewelry
        'chain', 'ring', 'necklace', 'bracelet'
    }

def separate_brands_and_items(brand_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate brand mentions DataFrame into actual brands and fashion items.

    This separation ensures:
    - Actual brands are sent to Google Trends API
    - Fashion items are analyzed via taxonomy table only

    Args:
        brand_results_df: DataFrame with 'brand_name' column containing both brands and items

    Returns:
        Tuple of (actual_brands_df, fashion_items_df)

    Example:
        Input: ["Nike", "Gucci", "watch", "bag", "t-shirt"]
        Output: (["Nike", "Gucci"], ["watch", "bag", "t-shirt"])
    """
    if brand_results_df.empty:
        return brand_results_df.copy(), pd.DataFrame()

    fashion_items = get_fashion_item_labels()

    # Case-insensitive separation
    is_fashion_item = brand_results_df['brand_name'].str.lower().isin(fashion_items)

    fashion_items_df = brand_results_df[is_fashion_item].copy()
    actual_brands_df = brand_results_df[~is_fashion_item].copy()

    return actual_brands_df, fashion_items_df


def _analyze_with_popularity_fallback(
    trends_service,
    popularity_analyzer,
    enriched_results: pd.DataFrame,
    brand_results_df: pd.DataFrame,
    brand_name: str
) -> list[dict]:
    """Try popularity analysis first, fallback to clustering then aggregate.

    Single Responsibility: Encapsulates the try-popularity-fallback-to-clustering logic.

    Args:
        trends_service: TrendsService instance
        popularity_analyzer: PopularityAnalyzer instance
        enriched_results: DataFrame with enriched lyrics
        brand_results_df: DataFrame with brand mentions
        brand_name: Brand to analyze

    Returns:
        List of trend analysis results
    """
    # Try popularity-based analysis first (viral song hypothesis)
    print(f"🎤 Trying popularity-based analysis for {brand_name}")
    popularity_result = popularity_analyzer.find_best_impact(
        enriched_results,
        brand_name,
        min_change=6.0
    )

    if popularity_result:
        # Found significant impact from popular song
        song_info = popularity_result['song']
        print(
            f"   ✓ Top hit: '{song_info['title']}' by {song_info['artist']} "
            f"(popularity: {song_info['popularity_weight']:.1f}, "
            f"impact: {popularity_result['impact']:+.1f}%)"
        )
        return [popularity_result]
    else:
        # Fallback to multi-year analysis
        print(f"   No significant impact from popular songs, trying multi-year analysis...")
        return trends_service.get_trends_by_mention_year(
            brands_df=brand_results_df,
            brand=brand_name
        )


def extract_artist_filter(question: str, conn) -> str | None:
    """Extract artist name via exact DB match against artists table."""
    import re
    # Find capitalized phrases (potential artist names)
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)

    for phrase in words:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT artist_name FROM artists WHERE LOWER(artist_name) = LOWER(%s)",
                [phrase]
            )
            result = cur.fetchone()
            if result:
                return result[0]
    return None

async def parallel_search(vs: VectorStore, question: str, artist_filter: str,
                         start_date: str, end_date: str, filter_brands_by_artist: bool = True):
    """Execute 4 searches concurrently using thread pool.

    Args:
        filter_brands_by_artist: If False, don't apply artist filter to brand_mentions.
                                 This is useful for comparative queries where we want all
                                 brand mentions, not just from one artist.
    """
    loop = asyncio.get_event_loop()

    # For comparative queries, don't filter brand_mentions by artist
    brand_artist_filter = artist_filter if filter_brands_by_artist else None

    tasks = [
        loop.run_in_executor(None, vs.search_with_joins,
                           question, 30, start_date, end_date, True, brand_artist_filter),
        loop.run_in_executor(None, vs.search,
                           question, 40, None, True, "enriched_lyrics", artist_filter),
        loop.run_in_executor(None, vs.search,
                           question, 20, None, True, "full_lyrics", artist_filter),
        loop.run_in_executor(None, vs.search_taxonomy,
                           question, 5, True)
    ]

    results = await asyncio.gather(*tasks)
    return {
        'brand_mentions': results[0],
        'enriched_lyrics': results[1],
        'full_lyrics': results[2],
        'taxonomy': results[3]
    }
    
def extract_fashion_terms(lyrics_df, client) -> str:
    """Extract fashion items from lyrics using LLM."""
    if lyrics_df.empty:
        return ""
    
    sample = lyrics_df['contents'].head(10).str.cat(sep='\n')[:3000]
    
    terms = client.chat.completions.create(
        model="gpt-5",
        response_model=FashionTerms,
        messages=[{
            "role": "system",
            "content": "Extract fashion items/garments mentioned: leather, denim, sneakers, boots, jackets, etc. Return comma-separated list."
        }, {
            "role": "user",
            "content": f"Extract fashion terms from:\n{sample}"
        }]
    )
    
    return ", ".join(terms.items) if terms.items else ""

def extract_date_range(query: str) -> tuple[str, str]:
    """Extract start/end dates from query text."""
    
    # Handle "March 15, 2023" or "March 15th, 2023"
    full_date = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', 
        query, 
        re.IGNORECASE
    )
    if full_date:
        month_name, day, year = full_date.groups()
        date_obj = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
        if 'after' in query.lower() or 'spike' in query.lower():
            end_date = date_obj + timedelta(days=14)
            return date_obj.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d'), date_obj.strftime('%Y-%m-%d')
    
    # Handle "November 2022" (month-year)
    month_year = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 
        query, 
        re.IGNORECASE
    )
    if month_year:
        month_name, year = month_year.groups()
        month_num = datetime.strptime(month_name, "%B").month
        last_day = calendar.monthrange(int(year), month_num)[1]
        return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day}"
    
    # Handle year ranges
    range_patterns = [
        r'(\d{4})\s*-\s*(\d{4})',
        r'from\s+(\d{4})\s+to\s+(\d{4})',
        r'between\s+(\d{4})\s*-\s*(\d{4})',
        r'from\s+(\d{4})\s+through\s+(\d{4})'
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            start, end = match.groups()
            return f"{start}-01-01", f"{end}-12-31"
    
    # Handle single year
    single_year = re.search(r'(?:in|from|year|of)\s+(\d{4})\b', query, re.IGNORECASE)
    if single_year:
        year = single_year.group(1)
        return f"{year}-01-01", f"{year}-12-31"
    
    return None, None

def parse_release_date(date_str: str) -> datetime | None:
    """Parse various date formats from database to datetime."""
    if not date_str or date_str == 'Unknown':
        return None
    
    try:
        return datetime.strptime(str(date_str), '%m/%d/%Y')
    except:
        pass
    
    try:
        return datetime.strptime(str(date_str), '%Y-%m-%d')
    except:
        pass
    
    try:
        return datetime.strptime(str(date_str), '%Y')
    except:
        pass
    
    return None

def cluster_mentions_by_month(results):
    """Group mentions by month, return {month: [dates]}."""
    monthly_clusters = defaultdict(list)
    
    for date_str in results['release_date'].dropna().unique():
        parsed = parse_release_date(date_str)
        if parsed:
            month_key = parsed.strftime('%Y-%m')
            monthly_clusters[month_key].append(parsed)
    
    for month in monthly_clusters:
        monthly_clusters[month] = sorted(monthly_clusters[month])
    
    return dict(monthly_clusters)

def query_system(question: str, limit: int = 50):
    print(f"\n{'='*80}")
    print(f"Query: {question}")
    print(f"{'='*80}\n")
    
    start_date, end_date = extract_date_range(question)
    
    if start_date and end_date:
        print(f"📅 Date Range: {start_date} to {end_date}")
    
    client = instructor.from_openai(OpenAI())
    decision = client.chat.completions.create(
        model="gpt-5",
        response_model=TrendDecision,
        messages=[{
            "role": "system",
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
        }, {
            "role": "user",
            "content": question
        }]
    )

    aggregation_data = None
    if decision.use_sql_aggregation:
        print("📊 SQL Aggregation Query - Direct Database Counts")
        aggregator = SQLAggregator()
        aggregation_data = route_aggregation_query(question, decision, aggregator)
        print(f"  Type: {aggregation_data['aggregation_type']}")
        print(f"  Rows: {len(aggregation_data['aggregation_results'])}")
        print(aggregation_data['aggregation_results'].head(15).to_string(index=False))
    
    vs = VectorStore()
    artist_filter = extract_artist_filter(question, vs.conn)

    # Only skip artist filtering if query compares multiple artists
    # Single-artist-multi-brand queries (e.g., "Travis's luxury brands") should still filter by artist
    filter_brands = len(decision.artist_names) <= 1

    try:
        results_dict = asyncio.run(parallel_search(vs, question, artist_filter, start_date, end_date, filter_brands))

        brand_results = results_dict['brand_mentions']
        enriched_results = results_dict['enriched_lyrics']
        lyrics_results = results_dict['full_lyrics']
        taxonomy_results = results_dict['taxonomy']

        print(f"\n📊 Data Sources Found:")
        print(f"  ✓ Brand mentions: {len(brand_results)}")
        print(f"  ✓ Enriched lyrics: {len(enriched_results)}")
        print(f"  ✓ Full lyrics: {len(lyrics_results)}")
        print(f"  ✓ Taxonomy: {len(taxonomy_results)}")

        # Load full taxonomy for fashion trends analysis
        full_taxonomy = vs.load_full_taxonomy()

        # Analyze fashion trends if enriched_lyrics has data
        fashion_trends = []
        if not enriched_results.empty:
            trends_service = TrendsService()
            fashion_trends = trends_service.analyze_fashion_trends(
                enriched_lyrics_df=enriched_results,
                taxonomy_df=full_taxonomy,  # ✅ Fixed: use full taxonomy
                start_date=start_date or "2018-01-01",
                end_date=end_date or "2025-12-31"
            )
            
            if fashion_trends:
                print(f"\n👗 Fashion Trends Analysis:")
                print(f"  Found {len(fashion_trends)} significant fashion trends")
            else:
                print(f"\n👗 Fashion Trends Analysis:")
                print(f"  No significant fashion trends found")

        # Separate brands from fashion items (Google Trends vs Taxonomy)
        actual_brands_df, fashion_items_df = separate_brands_and_items(brand_results)

        if not fashion_items_df.empty:
            print(f"🔍 Separated {len(fashion_items_df)} fashion items from {len(actual_brands_df)} brand mentions")

        # Auto-trigger trends for multi-brand queries that weren't classified as comparative
        # Skip auto-trigger if query is asking about fashion items/clothing
        if (not decision.needs_trends and
            not decision.use_sql_aggregation and
            not actual_brands_df.empty and
            'items' not in question.lower() and
            'clothing' not in question.lower()):
            unique_brands = actual_brands_df['brand_name'].unique()
            if len(unique_brands) >= 2:
                print(f"🔄 Auto-triggering comparative trends ({len(unique_brands)} brands found)")
                decision.comparative_query = True
                decision.needs_trends = True

        trends_data_list = []

        if decision.needs_trends:
            if decision.comparative_query:
                print("📊 Comparative query detected")

                trend_start = start_date or "2022-01-01"
                trend_end = end_date or "2022-12-31"

                trends_service = TrendsService()
                
                # Your fix: fetch actual trend data for each brand
                # Filter out None values to prevent crash
                brand_names = [b for b in actual_brands_df['brand_name'].unique().tolist() if b is not None]

                # Filter to pre-computed data range (2010-2025)
                original_count = len(actual_brands_df)
                actual_brands_df['year'] = pd.to_datetime(actual_brands_df['release_date'], format='mixed').dt.year
                actual_brands_df = actual_brands_df[
                    (actual_brands_df['year'] >= PRECOMPUTED_START_YEAR) &
                    (actual_brands_df['year'] <= PRECOMPUTED_END_YEAR)
                ]
                filtered_count = original_count - len(actual_brands_df)
                if filtered_count > 0:
                    logger.info(f"   Filtered {filtered_count} mentions outside pre-computed range ({PRECOMPUTED_START_YEAR}-{PRECOMPUTED_END_YEAR})")

                if not brand_names:
                    logger.warning("All brand names are None in comparative query")
                    trends_data_list = []
                else:
                    trends_data_list = []

                    for brand_name in brand_names[:5]:  # Limit to top 5
                        brand_mentions = trends_service._filter_by_brand_case_insensitive(
                            actual_brands_df,
                            brand_name
                        )
                        mention_dates = trends_service._extract_mention_dates(brand_mentions)

                        if mention_dates:
                            # Create window from mention clustering if no dates or window too wide
                            brand_window_start = trend_start
                            brand_window_end = trend_end

                            # Always calculate anchor from peak cluster for consistent split point
                            mention_months = [datetime.fromisoformat(d).replace(day=1) for d in mention_dates]
                            peak_month = Counter(mention_months).most_common(1)[0][0]
                            cluster_dates = [d for d in mention_dates if datetime.fromisoformat(d).replace(day=1) == peak_month]
                            anchor = datetime.fromisoformat(min(cluster_dates))
                            split_date = anchor.isoformat()  # Always use anchor as split

                            if not start_date or (datetime.fromisoformat(trend_end) - datetime.fromisoformat(trend_start)).days > 90:
                                # Wide window or no user dates: create narrow window around anchor
                                brand_window_start = (anchor - timedelta(days=30)).strftime('%Y-%m-%d')
                                brand_window_end = (anchor + timedelta(days=60)).strftime('%Y-%m-%d')
                            # else: keep the user-specified trend_start/trend_end window

                            # Try pre-computed data first, fallback to API
                            try:
                                trends_data = trends_service.get_brand_trends_from_precomputed(
                                    brand=brand_name,
                                    start_date=brand_window_start,
                                    end_date=brand_window_end,
                                    mention_dates=split_date  # Pass only split date, not all dates
                                )
                            except BrandNotFoundError:
                                logger.warning(f"Brand '{brand_name}' not in pre-computed data, using API")
                                trends_data = trends_service.get_brand_trends(
                                    TrendsRequest(
                                        brand=brand_name,
                                        start_date=brand_window_start,
                                        end_date=brand_window_end
                                    ),
                                    mention_dates=split_date
                                )
                            trends_data_list.append({
                                'brand': brand_name,
                                'mention_count': len(brand_mentions),
                                'trends': trends_data
                            })

                print(f"🎯 Analyzed {len(trends_data_list)} brands")
                for item in trends_data_list:
                    print(f"  ✓ {item['brand']}: {item['trends'].percent_change:+.1f}% change")

            elif decision.needs_trends and not decision.brand:
                # Extract top mentioned brand and analyze it
                brand_names = actual_brands_df['brand_name'].unique().tolist()
                if brand_names:
                    decision.brand = brand_names[0]  # Use top brand
                    print(f"🎯 Auto-selected top brand: {decision.brand}")
                    # Continue with existing single-brand logic
                    trends_service = TrendsService()
                    popularity_analyzer = PopularityAnalyzer(trends_service)

                    # Use helper function for popularity analysis with fallback
                    trends_data_list = _analyze_with_popularity_fallback(
                        trends_service,
                        popularity_analyzer,
                        enriched_results,
                        actual_brands_df,
                        decision.brand
                    )

                    # Final fallback: if no yearly trends found
                    if not trends_data_list:
                        print(f"🔄 No yearly trends found, using aggregate analysis...")
                        mention_dates = trends_service._extract_mention_dates(actual_brands_df)
                        if mention_dates:
                            logger.info(f"📊 Aggregate analysis for {decision.brand}")
                            logger.info(f"   Total mentions: {len(mention_dates)}")
                            logger.info(f"   Date range: {min(mention_dates)} to {max(mention_dates)}")

                            # Find densest cluster
                            mention_months = [datetime.fromisoformat(d).replace(day=1) for d in mention_dates]
                            peak_month = Counter(mention_months).most_common(1)[0][0]
                            cluster_dates = [d for d in mention_dates if datetime.fromisoformat(d).replace(day=1) == peak_month]

                            # Anchor on earliest mention in peak cluster
                            anchor = datetime.fromisoformat(min(cluster_dates))
                            split_date = anchor.isoformat()  # Always use anchor as split point

                            # Window expansion: If user specified dates, expand for baseline + lag
                            if start_date and end_date:
                                user_start = datetime.fromisoformat(start_date)
                                user_end = datetime.fromisoformat(end_date)
                                window_start = (user_start - timedelta(days=28)).strftime('%Y-%m-%d')
                                window_end = (user_end + timedelta(days=28)).strftime('%Y-%m-%d')
                                # Keep split_date = anchor (don't override with start_date)
                                logger.info(f"📅 User date range: {start_date} to {end_date}")
                                logger.info(f"📊 Expanded trends window: {window_start} to {window_end}")
                                logger.info(f"🎯 Split point (anchor): {split_date}")
                                print(f"📅 Expanded user window {start_date} to {end_date} → {window_start} to {window_end} (for baseline)")
                            else:
                                window_start = (anchor - timedelta(days=30)).strftime('%Y-%m-%d')
                                window_end = (anchor + timedelta(days=60)).strftime('%Y-%m-%d')
                                logger.info(f"📊 Calculated window: {window_start} to {window_end}")
                        else:
                            # Default to recent 2-year window
                            window_start = start_date or "2022-01-01"
                            window_end = end_date or "2024-12-31"
                            split_date = window_start  # Use window start as split

                        logger.info(f"   Trends window: {window_start} to {window_end}")
                        logger.info(f"   Split point: {split_date}")

                        # Try pre-computed data first, fallback to API
                        try:
                            trends_data = trends_service.get_brand_trends_from_precomputed(
                                brand=decision.brand,
                                start_date=window_start,
                                end_date=window_end,
                                mention_dates=split_date  # Pass only the split date, not all dates
                            )
                        except BrandNotFoundError:
                            logger.warning(f"Brand '{decision.brand}' not in pre-computed data, using API")
                            trends_data = trends_service.get_brand_trends(
                                TrendsRequest(
                                    brand=decision.brand,
                                    start_date=window_start,
                                    end_date=window_end
                                ),
                                mention_dates=split_date
                            )
                        trends_data_list = [{
                            'month': 'aggregate',
                            'mention_count': len(actual_brands_df),
                            'trends': trends_data
                        }]

            elif decision.brand:
                print(f"📈 Fetching trends for {decision.brand}...")

                trends_service = TrendsService()
                popularity_analyzer = PopularityAnalyzer(trends_service)

                # Use helper function for popularity analysis with fallback
                trends_data_list = _analyze_with_popularity_fallback(
                    trends_service,
                    popularity_analyzer,
                    enriched_results,
                    actual_brands_df,
                    decision.brand
                )

                # Final fallback: if no yearly trends found but query explicitly asks
                if not trends_data_list:
                    print(f"🔄 No yearly trends found, using aggregate analysis...")

                    # When user specifies dates, ALWAYS expand window for baseline
                    if start_date and end_date:
                        window_start = (datetime.fromisoformat(start_date) - timedelta(days=28)).strftime('%Y-%m-%d')
                        window_end = (datetime.fromisoformat(end_date) + timedelta(days=28)).strftime('%Y-%m-%d')
                        split_date = start_date  # User's date = intervention point
                        logger.info(f"📅 User date range → trend window: {window_start} to {window_end}, split at {split_date}")
                        print(f"📅 Analyzing trends around {start_date}: {window_start} to {window_end}")
                    else:
                        # Fallback: use mention-based anchor
                        mention_dates = trends_service._extract_mention_dates(actual_brands_df)
                        if not mention_dates:
                            trends_data_list = []  # No mentions, no date range = skip trends

                        logger.info(f"📊 Aggregate analysis for {decision.brand}")
                        logger.info(f"   Total mentions: {len(mention_dates)}")
                        logger.info(f"   Date range: {min(mention_dates)} to {max(mention_dates)}")

                        # Find densest cluster
                        mention_months = [datetime.fromisoformat(d).replace(day=1) for d in mention_dates]
                        peak_month = Counter(mention_months).most_common(1)[0][0]
                        cluster_dates = [d for d in mention_dates if datetime.fromisoformat(d).replace(day=1) == peak_month]

                        # Anchor on earliest mention in peak cluster
                        anchor = datetime.fromisoformat(min(cluster_dates))
                        split_date = anchor.isoformat()
                        window_start = (anchor - timedelta(days=30)).strftime('%Y-%m-%d')
                        window_end = (anchor + timedelta(days=60)).strftime('%Y-%m-%d')
                        logger.info(f"📊 Calculated window: {window_start} to {window_end}")

                    # Ensure split point is within window with sufficient pre/post baseline
                    split_dt = datetime.fromisoformat(split_date)
                    window_start_dt = datetime.fromisoformat(window_start)
                    window_end_dt = datetime.fromisoformat(window_end)

                    if split_dt <= window_start_dt:
                        # Put split 1/3 into window to ensure pre-baseline
                        window_days = (window_end_dt - window_start_dt).days
                        split_dt = window_start_dt + timedelta(days=window_days // 3)
                        split_date = split_dt.isoformat()
                        logger.warning(f"⚠️  Split {split_date[:10]} before window start, adjusted to {split_dt.date()}")

                    logger.info(f"   Trends window: {window_start} to {window_end}")
                    logger.info(f"   Split point: {split_date}")

                    # Try pre-computed data first, fallback to API
                    try:
                        trends_data = trends_service.get_brand_trends_from_precomputed(
                            brand=decision.brand,
                            start_date=window_start,
                            end_date=window_end,
                            mention_dates=split_date  # Pass only the split date, not all dates
                        )
                    except BrandNotFoundError:
                        logger.warning(f"Brand '{decision.brand}' not in pre-computed data, using API")
                        trends_data = trends_service.get_brand_trends(
                            TrendsRequest(
                                brand=decision.brand,
                                start_date=window_start,
                                end_date=window_end
                            ),
                            mention_dates=split_date
                        )
                    trends_data_list = [{
                        'month': 'aggregate',
                        'mention_count': len(actual_brands_df),
                        'trends': trends_data
                    }]

                print(f"📊 Found {len(trends_data_list)} trend analyses")

                for item in trends_data_list:
                    month = item.get('month', 'aggregate')  # Default to 'aggregate' if missing
                    count = item.get('mention_count', 0)     # Default to 0 if missing
                    trends = item.get('trends')             # Will be None if missing

                    if month == 'aggregate':
                        print(f"  📊 Aggregate analysis: {count} total mentions")
                    else:
                        print(f"  📅 {month}: {count} mentions")

                    if trends and hasattr(trends, 'pre_mention_avg') and trends.pre_mention_avg > 0:
                        print(f"     Pre: {trends.pre_mention_avg}, Post: {trends.post_mention_avg}, Change: {trends.percent_change:+.1f}%")
        
        actual_brands_df = deduplicate_brand_mentions(actual_brands_df)

        synth = FashionSynthesizer()
        insight = synth.generate_insight(
            question,
            actual_brands_df,  # Only actual brands (fashion items in fashion_trends)
            enriched_results,
            lyrics_results,
            taxonomy_results,
            trends_data_list,
            decision.comparative_query,
            None,  # category_baseline removed
            fashion_trends,  # Fashion items passed here only
            aggregation_data
        )

        print(f"\n{insight.summary}\n")
        for finding in insight.key_findings:
            print(f"• {finding}")
        print(f"\nQuality: {insight.data_quality}\n")

        # Display brand mentions
        if not actual_brands_df.empty:
            unique_artists = actual_brands_df['artist_name'].unique()
            print(f"Artists: {', '.join([str(a) for a in unique_artists if a])}\n")

            recent = actual_brands_df.sort_values('release_date', ascending=False).head(10)
            print("Sample brand mentions:")
            
            # Check if we have metadata columns to display
            display_cols = ['artist_name', 'song_title', 'release_date']
            if 'popularity_weight' in recent.columns:
                display_cols.append('popularity_weight')
            print(recent[display_cols].to_string(index=False))

        # Display fashion item mentions separately (for debugging/transparency)
        if not fashion_items_df.empty:
            print(f"\nFashion items mentioned (analyzed via taxonomy):")
            recent_items = fashion_items_df.sort_values('release_date', ascending=False).head(5)
            # Rename for clarity - these are fashion items, not brands
            display_items = recent_items.rename(columns={'brand_name': 'item_name'})
            
            # Check if we have metadata columns to display
            display_cols = ['artist_name', 'song_title', 'item_name', 'release_date']
            if 'popularity_weight' in display_items.columns:
                display_cols.append('popularity_weight')
            print(display_items[display_cols].to_string(index=False))

        print(f"\n{'='*80}\n")

        vs.close()

    except Exception as e:
        print(f"✗ Error during search: {str(e)}")
        raise

if __name__ == "__main__":
    queries = [
        # set 1
        # # Fix ambiguous queries from failures
        # "Did Nike spike after Drake's Dark Lane Demo Tapes in May 2020?",  # Explicit date fixes Query 6
        # # "What fashion items does Future mention?",  # Clear artist reference

        # # Test high-coverage artists
        # # "Compare Gucci vs Louis Vuitton in Lil Durk's lyrics",  # Durk has 30 brands
        # "What trends emerged after Future's DS2 album in 2015?",  # Future = 392 total coverage

        # # Edge cases
        # # "Prada performance after Gunna mentions in 2024",  # Recent data, balanced artist
        # "Did Yeezy spike after Kanye mentions?",  # Self-referential brand/artist
        
        # # Additional test queries
        # # "What fashion trends emerged after Playboi Carti's album I Am Music?",
        # # "Did Gucci spike after Cardi B's 7 mentions in April 2018?",
        # # "Who drove bigger brand impact in 2018: Travis Scott or Cardi B?",
        # # "What fashion items does Playboi Carti mention in his lyrics?",
        # # "List clothing Future references",
        # # "What items does Gucci mention in lyrics?",
        # "Which artist drove bigger impact?",
        # "Did Maybach spike after DaBaby mentions in 2022?",

        # set 2
        # # Temporal analysis - specific albums
        #  "Did Gucci spike after Migos Culture album in January 2017?",
        # "Nike trends after Travis Scott's Astroworld in August 2018?",
        
        # # Comparative queries
        # "Compare Nike vs Adidas mentions across all artists and time periods",
        # "Compare Balenciaga vs Prada impact in 2023",
        
        # # Linguistic/retrieval depth
        # "What metaphors or symbolism appear when artists reference Louis Vuitton?",
        # "Show the variety of contexts in which Versace is mentioned across different artists",
        
        # # Aggregation queries
        # "For Kendrick Lamar, what are the top brands referenced across his discography?",
        # "Which songs contain the highest number of brand references?",
        
        # # Full synthesis
        # "Generate a ranked list of brands that benefit most from hip-hop cultural relevance",
        
        # # Multi-year comparative
        # "Which artists drove the biggest brand impact between 2015-2020?"

        # set 3
        # Trajectory/temporal
        # "What is the influence trajectory of Versace brand across the dataset over time?",
        
        # # Artist-brand relationships
        # "Which artists contribute the most to the visibility of a particular brand?",
        # "Suggest brands that fit an artist's lyrical identity based on their mention history. Give a list of 10 artists.",
        
        # # Brand-specific deep dive
        # "Generate a marketing insight report for 'Tom Ford' based on all mentions.",
        
        # # # Linguistic/symbolic analysis
        # # "What kinds of metaphors or symbolism appear when artists reference luxury brands?",
        
        # # # Cultural analysis
        # # "What does the dataset reveal about brand power hierarchy in hip-hop culture?",
        # # "What makes hip-hop music a powerful channel for brand endorsement? Support with dataset evidence.",
        
        # # # Full synthesis (best one)
        # # "Using all available data, generate a ranked list of brands that benefit the most from hip-hop cultural relevance, and explain why.",

        # # set 4
        # # Brand frequency/reach
        # "Which brands have appeared in the highest number of unique songs?",
        # "Which songs contain the highest number of brand references?",
        
        # # Artist-brand analysis
        # "Which brands are mentioned predominantly by luxury-focused artists vs. streetwear-focused artists?",
        # "For artist Kendrick Lamar, what are the top brands referenced across his discography?",
        # "Which artists have the most diverse brand vocabulary?",
        
        # # Context/semantic analysis
        # "For Gucci brand, what are the most common words or themes that appear in nearby context windows?",
        # "Show the variety of contexts in which Louis Vuitton brand is mentioned across different artists.",
        # "What are the top 20 most informational brand mentions from data? Provide context windows.",
        
        # # Comparative
        # "Compare the mentions of Nike vs Adidas across all artists and time periods.",
        # "Which brand dominates more in the 2010s vs 2000s in mentions?",
        
        # # Cultural/qualitative
        # "What are the most iconic or culturally influential brand mentions in the dataset?",
        # "How do brand references vary between emerging artists and established artists?",
        
        # # Advanced (may need new functionality)
        # "Construct a brand-co-occurrence graph: which brands appear together most often in hip-hop lyrics?",
        # "Which artists have the most similar brand profiles? Provide clusters formed by RAG.",

        # "Did Versace spike after Migos 'Versace' single in June 2013?",
        # "Did Gucci spike after Migos Culture album in January 2017?",
        # "What was the trend impact of Jay-Z's 'Tom Ford' single in July 2013?",
        # "Which artists have the most diverse brand vocabulary?",
        # # "Which songs contain the highest number of brand references?",

    ]
    
    for q in queries:
        query_system(q)