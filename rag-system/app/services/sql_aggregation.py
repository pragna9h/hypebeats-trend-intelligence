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