"""
Popularity-based trend analyzer.
Identifies high-impact songs and their trend windows.
"""
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
import logging

from app.services.trends_service import BrandNotFoundError
from app.models.trends import TrendsRequest

logger = logging.getLogger(__name__)


class PopularSong(BaseModel):
    """Represents a popular song with brand mention."""
    artist: str
    title: str
    release_date: datetime
    popularity_weight: float
    brand: str


class PopularityAnalyzer:
    """Analyzes trends around popular songs."""

    def __init__(self, trends_service):
        """Initialize with a trends service for data fetching.

        Args:
            trends_service: TrendsService instance for fetching trend data
        """
        self.trends_service = trends_service

    def get_top_songs(
        self,
        enriched_df: pd.DataFrame,
        brand: str,
        top_n: int = 3
    ) -> List[PopularSong]:
        """Extract top N songs by popularity weight.

        Args:
            enriched_df: DataFrame with enriched lyrics and metadata
            brand: Brand name being analyzed
            top_n: Number of top songs to return (default: 3)

        Returns:
            List of PopularSong objects sorted by popularity_weight (desc)
        """
        if enriched_df.empty:
            return []

        # CRITICAL: Filter to only rows with popularity_weight (skip NULL from brand_mentions/full_lyrics)
        enriched_df = enriched_df[enriched_df['popularity_weight'].notna()].copy()

        if enriched_df.empty:
            logger.info(f"   No songs with popularity_weight found for {brand}")
            return []

        songs = []
        for _, row in enriched_df.iterrows():
            # Skip songs with 'Unknown' or None title (bad data entries)
            title = row.get('song_title')
            if not title or title == 'Unknown':
                continue

            artist = row.get('artist_name', 'Unknown')

            # Skip songs without popularity_weight
            pop = row.get('popularity_weight')
            if pop is None or pd.isna(pop):
                continue

            try:
                # Parse release date safely
                release_date_str = row.get('release_date')
                if not release_date_str:
                    release_date_str = '2020-01-01'
                release_date = datetime.fromisoformat(str(release_date_str))

                songs.append(PopularSong(
                    artist=artist,
                    title=title,
                    release_date=release_date,
                    popularity_weight=float(pop),
                    brand=brand
                ))
            except Exception as e:
                logger.debug(f"Skipping song due to parsing error: {e}")
                continue

        # Sort by popularity (descending) and return top N
        sorted_songs = sorted(songs, key=lambda x: x.popularity_weight, reverse=True)[:top_n]

        # Log if no songs found
        if not sorted_songs:
            logger.info(f"   No songs with popularity_weight found (checked {len(enriched_df)} rows)")

        return sorted_songs

    def analyze_song_impact(
        self,
        song: PopularSong,
        pre_days: int = 30,
        post_days: int = 60
    ):
        """Get trend window around a specific song release.

        Args:
            song: PopularSong object to analyze
            pre_days: Days before release to include (default: 30)
            post_days: Days after release to include (default: 60)

        Returns:
            TrendsResponse object or None if analysis fails
        """
        start = (song.release_date - timedelta(days=pre_days)).strftime('%Y-%m-%d')
        end = (song.release_date + timedelta(days=post_days)).strftime('%Y-%m-%d')

        try:
            # Try pre-computed data first
            return self.trends_service.get_brand_trends_from_precomputed(
                brand=song.brand,
                start_date=start,
                end_date=end,
                mention_dates=[song.release_date.isoformat()]
            )
        except BrandNotFoundError:
            logger.warning(f"Brand '{song.brand}' not in pre-computed data, using API fallback")
            try:
                # Fallback to Google Trends API
                return self.trends_service.get_brand_trends(
                    request=TrendsRequest(
                        brand=song.brand,
                        start_date=start,
                        end_date=end
                    ),
                    mention_dates=[song.release_date.isoformat()]
                )
            except Exception as e:
                logger.error(f"Failed to get trends for {song.title} via API: {e}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get trends for {song.title}: {e}")
            return None

    def find_best_impact(
        self,
        enriched_df: pd.DataFrame,
        brand: str,
        min_change: float = 6.0
    ):
        """Find highest-impact song from top 3 popular songs.

        Args:
            enriched_df: DataFrame with enriched lyrics and metadata
            brand: Brand name to analyze
            min_change: Minimum percent change to consider significant (default: 10.0)

        Returns:
            Dict with 'song', 'trends', 'impact', 'mention_count' keys, or None
        """
        top_songs = self.get_top_songs(enriched_df, brand)

        if not top_songs:
            logger.info(f"   No songs with popularity_weight found for {brand}")
            return None

        logger.info(f"   Top {len(top_songs)} songs by popularity:")
        for i, song in enumerate(top_songs, 1):
            logger.info(f"     {i}. '{song.title}' by {song.artist} (pop: {song.popularity_weight:.1f})")

        results = []
        for song in top_songs:
            trend = self.analyze_song_impact(song)
            if trend and abs(trend.percent_change) >= min_change:
                results.append({
                    'song': song.dict(),  # Convert Pydantic model to dict
                    'trends': trend,
                    'impact': trend.percent_change,
                    'mention_count': 1  # Single song mention (for compatibility)
                })
                logger.info(
                    f"     → '{song.title}': {trend.percent_change:+.1f}% "
                    f"(pre: {trend.pre_mention_avg}, post: {trend.post_mention_avg})"
                )

        if not results:
            logger.info(f"   No songs showed significant impact (>{min_change}% threshold)")
            return None

        # Return song with highest absolute impact
        best = max(results, key=lambda x: abs(x['impact']))
        logger.info(f"   ✓ Best impact: '{best['song']['title']}' with {best['impact']:+.1f}% change")
        return best
