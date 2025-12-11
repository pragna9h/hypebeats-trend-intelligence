# app/services/trends_service.py
import logging
from pytrends.request import TrendReq
from openai import OpenAI
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import instructor
import pandas as pd
import time
import random
from sqlalchemy import create_engine, text

from app.models.trends import TrendsRequest, TrendsResponse, TrendsDataPoint
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class BrandNotFoundError(Exception):
    """Raised when a brand is not found in pre-computed data."""
    pass


# Pre-computed data coverage (brand_trends_monthly.csv)
PRECOMPUTED_START_YEAR = 2010
PRECOMPUTED_END_YEAR = 2025


class TrendsService:
    """Service for interacting with Google Trends API."""

    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        settings = get_settings()
        self.engine = create_engine(settings.database.service_url)

    def get_brand_trends(
        self, 
        request: TrendsRequest, 
        mention_dates: list[str] | str | None = None
    ) -> TrendsResponse:
        """Fetch and process Google Trends data for a brand.
        
        Args:
            request: TrendsRequest with brand, dates, geo
            mention_dates: Optional date(s) when brand was mentioned (ISO format)
                          Can be single string or list. Uses first date for split.
        """
        # Polite sleep + retry for rate limiting
        for attempt in range(3):
            try:
                # Proactive delay to prevent rate limiting (8-9s with jitter)
                time.sleep(8 + random.uniform(0, 1))

                self.pytrends.build_payload(
                    [request.brand],
                    timeframe=f'{request.start_date} {request.end_date}',
                    geo=request.geo
                )
                df = self.pytrends.interest_over_time()
                break  # Success - exit retry loop

            except Exception as e:
                if ('429' in str(e) or 'TooManyRequestsError' in str(e)) and attempt < 2:
                    wait = 60 * (attempt + 1)  # 60s, 120s exponential backoff
                    logger.warning(f"⚠️  Rate limit hit for {request.brand}, retrying in {wait}s... (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ Failed to fetch trends for {request.brand}: {e}")
                    raise

        if df.empty:
            raise ValueError(f"No trends data found for {request.brand}")
        
        data_points = [
            TrendsDataPoint(date=date, value=value)
            for date, value in df[request.brand].items()
        ]
        
        # Initialize pre/post metrics (defaults to 0)
        pre_avg = 0
        post_avg = 0
        pct_change = 0
        
        # Calculate pre/post if mention dates provided
        if mention_dates:
            # Handle both single date string and list of dates
            first_mention = mention_dates if isinstance(mention_dates, str) else mention_dates[0]

            mention_dt = datetime.fromisoformat(first_mention)
            pre_data = [p for p in data_points if p.date < mention_dt]
            post_data = [p for p in data_points if p.date >= mention_dt]

            if pre_data and post_data:
                pre_avg = sum(p.value for p in pre_data) / len(pre_data)
                post_avg = sum(p.value for p in post_data) / len(post_data)
                if pre_avg > 0:
                    pct_change = ((post_avg - pre_avg) / pre_avg) * 100

            # Detailed logging for diagnostics
            logger.info(f"📈 {request.brand} trend calculation:")
            logger.info(f"   Total data points: {len(data_points)}")
            logger.info(f"   Pre-mention points: {len(pre_data)} | Avg: {pre_avg:.2f}")
            logger.info(f"   Post-mention points: {len(post_data)} | Avg: {post_avg:.2f}")
            logger.info(f"   Change: {pct_change:+.1f}%")

            # Validation warnings
            if len(pre_data) < 4:
                logger.warning(f"⚠️  Insufficient pre-baseline: only {len(pre_data)} weeks")
            if len(post_data) < 4:
                logger.warning(f"⚠️  Insufficient post-period: only {len(post_data)} weeks")

        return TrendsResponse(
            brand=request.brand,
            timeframe=f"{request.start_date} to {request.end_date}",
            data=data_points,
            average_interest=round(df[request.brand].mean(), 2),
            related_topics=[],
            pre_mention_avg=round(pre_avg, 2),
            post_mention_avg=round(post_avg, 2),
            percent_change=round(pct_change, 2)
        )

    def _extract_mention_dates(self, brand_df: pd.DataFrame) -> list[str] | None:
        """Extract and parse mention dates from brand results DataFrame.

        Args:
            brand_df: DataFrame with 'release_date' column containing date strings

        Returns:
            Sorted list of ISO format dates, or None if no valid dates found
        """
        mention_dates = []

        for date_str in brand_df['release_date'].dropna().unique():
            parsed = self._parse_release_date(date_str)
            if parsed:
                mention_dates.append(parsed.isoformat())

        return sorted(mention_dates) if mention_dates else None

    def _parse_release_date(self, date_str: str) -> datetime | None:
        """Parse various date formats from database to datetime."""
        if not date_str or date_str == 'Unknown':
            return None

        # Try MM/DD/YYYY
        try:
            return datetime.strptime(str(date_str), '%m/%d/%Y')
        except:
            pass

        # Try YYYY-MM-DD
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            pass

        # Try YYYY only
        try:
            return datetime.strptime(str(date_str), '%Y')
        except:
            pass

        return None

    def _filter_by_brand_case_insensitive(
        self,
        df: pd.DataFrame,
        brand_name: str,
        column: str = 'brand_name'
    ) -> pd.DataFrame:
        """Filter DataFrame by brand name (case-insensitive).

        Single Responsibility: Handle case-insensitive brand filtering.

        Args:
            df: DataFrame with brand column
            brand_name: Brand name to filter by
            column: Column name containing brand (default: 'brand_name')

        Returns:
            Filtered DataFrame (empty if brand_name is None)
        """
        if not brand_name or df.empty or column not in df.columns:
            return pd.DataFrame()

        return df[df[column].str.lower() == brand_name.lower()]

    def get_comparative_trends(
        self,
        brand_results_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        min_mentions: int = 1,
        max_brands: int = 3
    ) -> dict:
        """Analyze Google Trends for top brands from search results.

        Args:
            brand_results_df: DataFrame with brand mentions (must have 'brand_name', 'release_date')
            start_date: Trend analysis start date (YYYY-MM-DD)
            end_date: Trend analysis end date (YYYY-MM-DD)
            min_mentions: Minimum mentions required for a brand to be analyzed
            max_brands: Maximum number of brands to analyze

        Returns:
            Dict with 'brands' (list of brand trend dicts)
        """
        if brand_results_df.empty:
            return {'brands': []}

        # Count brand mentions and filter by threshold
        brand_counts = brand_results_df['brand_name'].value_counts()
        top_brands = brand_counts[brand_counts >= min_mentions].index[:max_brands].tolist()

        trends_data_list = []

        for brand in top_brands:
            try:
                brand_data = self._filter_by_brand_case_insensitive(brand_results_df, brand)
                mention_dates = self._extract_mention_dates(brand_data)

                trends_data = self.get_brand_trends(
                    request=TrendsRequest(
                        brand=brand,
                        start_date=start_date,
                        end_date=end_date
                    ),
                    mention_dates=mention_dates
                )

                trends_data_list.append({
                    'brand': brand,
                    'mention_count': len(brand_data),
                    'trends': trends_data
                })

            except Exception as e:
                # Continue analyzing other brands if one fails
                print(f"  ⚠️  {brand} trends failed: {e}")
                continue

        return {
            'brands': trends_data_list
        }

    def analyze_fashion_trends(
        self,
        enriched_lyrics_df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> list[dict]:
        """Analyze fashion item mentions against taxonomy baselines.

        Returns top 5 items by impact, grouped by category (2+ mentions).
        """
        if enriched_lyrics_df.empty or taxonomy_df.empty:
            return []

        # Extract canonical_labels from enriched_lyrics (handle flattened structure)
        items = []
        for _, row in enriched_lyrics_df.iterrows():
            # Handle both flattened columns and nested metadata
            if 'canonical_label' in row and pd.notna(row['canonical_label']):
                label = row['canonical_label']
                date_str = row.get('release_date')
            elif 'metadata' in row and isinstance(row['metadata'], dict):
                metadata = row['metadata']
                label = metadata.get('canonical_label')
                date_str = metadata.get('release_date')
            else:
                continue
                
            if label:
                items.append({'label': label, 'date': date_str})

        if not items:
            return []

        # Map enriched labels to taxonomy canonical_labels
        LABEL_MAP = {
            'bag': 'tote bag',
            't-shirt': 'graphic tee',
            'tee': 'graphic tee',
            'shirt': 'flannel shirt',
            'heels': 'platform shoes',
            'stilettos': 'platform shoes',
            'boots': 'combat boots',
            'cap': 'baseball cap',
            'sweater': 'knit sweater',
            'jacket': 'fleece jacket',
            'jeans': 'mom jeans',
            'slides': 'slides',  # Direct mapping to taxonomy
            'flip flops': 'slides',  # Normalize flip flops variant
            'chain': None,  # Jewelry not in taxonomy
            'ring': None,
        }
        WATCH_LABELS = {'watch', 'ap', 'patek', 'rolex', 'patek philippe', 
                        'audemars piguet', 'cartier'}

        # Normalize labels to taxonomy + filter watches/jewelry
        for item in items:
            label = item['label'].lower()
            
            if label in WATCH_LABELS:
                item['label'] = None  # Tracked as brands
            elif label in LABEL_MAP:
                item['label'] = LABEL_MAP[label]
            # Else: keep original (try exact match)

        items = [i for i in items if i['label'] is not None]

        # Count occurrences
        from collections import Counter
        item_counts = Counter([item['label'].lower() for item in items])

        # Filter: 2+ mentions only
        significant_items = {k: v for k, v in item_counts.items() if v >= 2}

        results = []

        for item_name, count in significant_items.items():
            # Match to taxonomy (exact match works after normalization)
            taxonomy_match = taxonomy_df[
                taxonomy_df['canonical_label'].str.lower() == item_name
            ]

            if taxonomy_match.empty:
                continue

            row = taxonomy_match.iloc[0]
            monthly_trends = row['monthly_trends']
            category = row['category']

            # Get earliest mention date
            mention_dates = []
            for item in items:
                if item['label'].lower() == item_name:
                    parsed = self._parse_release_date(item['date'])
                    if parsed:
                        mention_dates.append(parsed)

            if not mention_dates:
                continue

            first_mention = min(mention_dates)

            # Calculate 1 month before, 2 months after
            pre_month = (first_mention - timedelta(days=30)).strftime('%Y-%m-01')
            post_month1 = first_mention.strftime('%Y-%m-01')
            post_month2 = (first_mention + timedelta(days=60)).strftime('%Y-%m-01')

            pre_val = None
            post_vals = []

            for trend in monthly_trends:
                month_str = trend['month'][:10]

                if month_str == pre_month:
                    pre_val = trend['interest']
                elif month_str in [post_month1, post_month2]:
                    post_vals.append(trend['interest'])

            if pre_val and post_vals:
                post_avg = sum(post_vals) / len(post_vals)
                pct_change = ((post_avg - pre_val) / pre_val) * 100 if pre_val > 0 else 0

                results.append({
                    'item': item_name,
                    'category': category,
                    'mention_count': int(count),
                    'pre_baseline': round(pre_val, 1),
                    'post_baseline': round(post_avg, 1),
                    'percent_change': round(pct_change, 1)
                })

        # Sort by mention count DESC, then percent change DESC
        results.sort(key=lambda x: (x['mention_count'], x['percent_change']), reverse=True)

        # Group by category, take top 2 per category
        from collections import defaultdict
        by_category = defaultdict(list)
        for r in results:
            by_category[r['category']].append(r)

        final = []
        for category in sorted(by_category.keys()):
            final.extend(by_category[category][:2])

        return final[:5]  # Top 5 overall

    def get_monthly_cluster_trends(
        self,
        brand_results_df: pd.DataFrame,
        brand_name: str,
        min_mentions: int = 3
    ) -> list[dict]:
        """Analyze trends for monthly clusters of brand mentions.

        Args:
            brand_results_df: DataFrame with brand mentions
            brand_name: Brand to analyze
            min_mentions: Minimum mentions per month to trigger analysis

        Returns:
            List of dicts with 'month', 'mention_count', 'trends' keys
        """
        if brand_results_df.empty:
            return []

        # Cluster mentions by month
        monthly_clusters = self._cluster_by_month(brand_results_df)

        # Filter for significant months
        significant_months = {
            month: dates for month, dates in monthly_clusters.items()
            if len(dates) >= min_mentions
        }

        trends_data_list = []

        for month, dates in sorted(significant_months.items()):
            try:
                first_date = dates[0]
                window_start = (first_date - timedelta(days=30)).strftime('%Y-%m-%d')
                window_end = (first_date + timedelta(days=60)).strftime('%Y-%m-%d')

                # Try pre-computed data first, fallback to API
                try:
                    trends_data = self.get_brand_trends_from_precomputed(
                        brand=brand_name,
                        start_date=window_start,
                        end_date=window_end,
                        mention_dates=[d.isoformat() for d in dates]
                    )
                except BrandNotFoundError:
                    logger.warning(f"Brand '{brand_name}' not in pre-computed data, using API")
                    trends_data = self.get_brand_trends(
                        request=TrendsRequest(
                            brand=brand_name,
                            start_date=window_start,
                            end_date=window_end
                        ),
                        mention_dates=[d.isoformat() for d in dates]
                    )

                trends_data_list.append({
                    'month': month,
                    'mention_count': len(dates),
                    'trends': trends_data
                })

            except Exception as e:
                print(f"  ⚠️  {month} trends failed: {e}")
                continue

        return trends_data_list

    def get_trends_by_mention_year(
        self,
        brands_df: pd.DataFrame,
        brand: str
    ) -> list[dict]:
        """Fetch separate trend windows for each year with mentions.

        Replaces aggregate analysis which loses multi-year data by using midpoint window.
        Groups mentions by year and analyzes each year independently.

        Args:
            brands_df: DataFrame with brand mentions
            brand: Brand name to analyze

        Returns:
            List of dicts sorted by absolute impact (descending):
            [{'year': 2016, 'mention_count': 5, 'trends': TrendsResponse, 'impact': 23.4}, ...]
        """
        brand_lower = brand.lower() if brand else None
        if not brand_lower:
            logger.warning("No brand specified for multi-year analysis")
            return []

        # Filter to brand
        brands_filtered = self._filter_by_brand_case_insensitive(brands_df, brand)
        if brands_filtered.empty:
            logger.info(f"   No mentions found for {brand}")
            return []

        # Group by year
        brands_filtered = brands_filtered.copy()
        brands_filtered['year'] = pd.to_datetime(brands_filtered['release_date'], format='mixed').dt.year

        results = []
        for year, group in brands_filtered.groupby('year'):
            # Skip years outside pre-computed data range
            if year < PRECOMPUTED_START_YEAR or year > PRECOMPUTED_END_YEAR:
                logger.info(f"   Skipping {year} (outside pre-computed data range {PRECOMPUTED_START_YEAR}-{PRECOMPUTED_END_YEAR})")
                continue

            dates = pd.to_datetime(group['release_date'], format='mixed')

            # Window: 30 days before first mention to 60 days after last
            start = (dates.min() - timedelta(days=30)).strftime('%Y-%m-%d')
            end = (dates.max() + timedelta(days=60)).strftime('%Y-%m-%d')

            try:
                trends = self.get_brand_trends_from_precomputed(
                    brand=brand_lower,
                    start_date=start,
                    end_date=end,
                    mention_dates=dates.dt.strftime('%Y-%m-%d').tolist()
                )

                if trends and trends.data:
                    results.append({
                        'year': int(year),
                        'mention_count': len(group),
                        'trends': trends,
                        'impact': trends.percent_change
                    })
                    logger.info(f"   {year}: {len(group)} mentions, {trends.percent_change:+.1f}% change")

            except BrandNotFoundError:
                logger.warning(f"Brand '{brand}' not in pre-computed data for {year}, using API fallback")
                try:
                    trends = self.get_brand_trends(
                        request=TrendsRequest(
                            brand=brand_lower,
                            start_date=start,
                            end_date=end
                        ),
                        mention_dates=dates.dt.strftime('%Y-%m-%d').tolist()
                    )

                    if trends and trends.data:
                        results.append({
                            'year': int(year),
                            'mention_count': len(group),
                            'trends': trends,
                            'impact': trends.percent_change
                        })
                        logger.info(f"   {year}: {len(group)} mentions, {trends.percent_change:+.1f}% change")
                except Exception as e:
                    logger.warning(f"Failed to get trends for {brand} in {year} via API: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to get trends for {brand} in {year}: {e}")
                continue

        # Sort by absolute impact (highest first)
        return sorted(results, key=lambda x: abs(x['impact']), reverse=True)

    def _query_brand_monthly_trends(
        self,
        brand: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Query pre-computed monthly trends for a brand.

        Args:
            brand: Brand name (case-insensitive)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with columns: label, month, interest

        Raises:
            BrandNotFoundError: If brand not found in pre-computed data
        """
        query = text("""
            SELECT label, month, interest
            FROM brand_trends_monthly
            WHERE LOWER(label) = LOWER(:brand)
            AND month >= CAST(:start_date AS DATE)
            AND month <= CAST(:end_date AS DATE)
            ORDER BY month
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={'brand': brand, 'start_date': start_date, 'end_date': end_date}
            )

        if df.empty:
            raise BrandNotFoundError(f"Brand '{brand}' not found in pre-computed data for date range {start_date} to {end_date}")

        return df

    def _calculate_pre_post_metrics(
        self,
        data_points: list[TrendsDataPoint],
        mention_dates: list[str] | str | None
    ) -> tuple[float, float, float]:
        """Calculate pre/post mention averages and percent change.

        Args:
            data_points: List of trend data points
            mention_dates: Date(s) when brand was mentioned (ISO format)

        Returns:
            Tuple of (pre_avg, post_avg, percent_change)
        """
        pre_avg = 0
        post_avg = 0
        pct_change = 0

        if not mention_dates or not data_points:
            return pre_avg, post_avg, pct_change

        # Handle both single date string and list of dates
        first_mention = mention_dates if isinstance(mention_dates, str) else mention_dates[0]
        mention_dt = datetime.fromisoformat(first_mention)

        pre_data = [p for p in data_points if p.date < mention_dt]
        post_data = [p for p in data_points if p.date >= mention_dt]

        if pre_data and post_data:
            pre_avg = sum(p.value for p in pre_data) / len(pre_data)
            post_avg = sum(p.value for p in post_data) / len(post_data)
            if pre_avg > 0:
                pct_change = ((post_avg - pre_avg) / pre_avg) * 100

        return pre_avg, post_avg, pct_change

    def get_brand_trends_from_precomputed(
        self,
        brand: str,
        start_date: str,
        end_date: str,
        mention_dates: list[str] | str | None = None
    ) -> TrendsResponse:
        """Fetch brand trends from pre-computed monthly data (no API call).

        Args:
            brand: Brand name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            mention_dates: Optional date(s) when brand was mentioned (ISO format)

        Returns:
            TrendsResponse with monthly aggregated data

        Raises:
            BrandNotFoundError: If brand not found in pre-computed data
        """
        # Round dates to nearest month boundaries for better coverage
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Round to first day of month for start, last day of month for end
        start_rounded = start_dt.replace(day=1).strftime('%Y-%m-%d')
        end_rounded = (end_dt.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        end_rounded = end_rounded.strftime('%Y-%m-%d')

        # Query pre-computed data
        monthly_df = self._query_brand_monthly_trends(brand, start_rounded, end_rounded)

        # Convert to TrendsDataPoint format
        data_points = [
            TrendsDataPoint(
                date=pd.to_datetime(row['month']).to_pydatetime(),
                value=int(row['interest'])
            )
            for _, row in monthly_df.iterrows()
        ]

        # Calculate pre/post metrics
        pre_avg, post_avg, pct_change = self._calculate_pre_post_metrics(
            data_points,
            mention_dates
        )

        # Log diagnostics
        if mention_dates:
            first_mention = mention_dates if isinstance(mention_dates, str) else mention_dates[0]
            mention_dt = datetime.fromisoformat(first_mention)
            pre_data = [p for p in data_points if p.date < mention_dt]
            post_data = [p for p in data_points if p.date >= mention_dt]

            logger.info(f"📈 {brand} trend calculation (pre-computed):")
            logger.info(f"   Total data points: {len(data_points)}")
            logger.info(f"   Pre-mention points: {len(pre_data)} | Avg: {pre_avg:.2f}")
            logger.info(f"   Post-mention points: {len(post_data)} | Avg: {post_avg:.2f}")
            logger.info(f"   Change: {pct_change:+.1f}%")

        return TrendsResponse(
            brand=brand,
            timeframe=f"{start_date} to {end_date}",
            data=data_points,
            average_interest=round(monthly_df['interest'].mean(), 2),
            related_topics=[],
            pre_mention_avg=round(pre_avg, 2),
            post_mention_avg=round(post_avg, 2),
            percent_change=round(pct_change, 2)
        )

    def _cluster_by_month(self, results_df: pd.DataFrame) -> dict[str, list[datetime]]:
        """Group mentions by month, return {month: [dates]}."""
        from collections import defaultdict
        monthly_clusters = defaultdict(list)

        for date_str in results_df['release_date'].dropna().unique():
            parsed = self._parse_release_date(date_str)
            if parsed:
                month_key = parsed.strftime('%Y-%m')
                monthly_clusters[month_key].append(parsed)

        # Sort dates within each month
        for month in monthly_clusters:
            monthly_clusters[month] = sorted(monthly_clusters[month])

        return dict(monthly_clusters)

class TrendDecision(BaseModel):
    needs_trends: bool
    use_sql_aggregation: bool = False  # True for count/frequency/ranking queries
    brand: str | None
    artist_names: list[str] = Field(default_factory=list)
    start_date: str | None
    end_date: str | None
    comparative_query: bool = False

def test_with_llm():
    client = instructor.from_openai(OpenAI())
    service = TrendsService()
    
    queries = [
        "Show Nike trends in January 2023",
        "Which brands does Drake mention?",
        "What was Gucci's popularity in 2022?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}\nQuery: {query}\n{'-'*60}")
        
        decision = client.chat.completions.create(
            model="gpt-5",
            response_model=TrendDecision,
            messages=[{
                "role": "system",
                "content": "Determine if query needs Google Trends data and extract brand/dates."
            }, {
                "role": "user",
                "content": query
            }]
        )
        
        print(f"Needs trends: {decision.needs_trends}")
        
        if decision.needs_trends and decision.brand:
            start = decision.start_date or "2023-01-01"
            end = decision.end_date or "2023-01-31"
            
            result = service.get_brand_trends(TrendsRequest(
                brand=decision.brand,
                start_date=start,
                end_date=end
            ))
            print(f"Brand: {result.brand}")
            print(f"Avg interest: {result.average_interest}")
            print(f"Related topics: {', '.join(result.related_topics) if result.related_topics else 'None'}")

if __name__ == "__main__":
    test_with_llm()