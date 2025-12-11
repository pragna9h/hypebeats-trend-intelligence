"""Unit tests for PopularityAnalyzer module."""
import pytest
from datetime import datetime
import pandas as pd
from app.services.popularity_analyzer import PopularityAnalyzer, PopularSong


@pytest.fixture
def sample_enriched_df():
    """Sample enriched lyrics DataFrame with 3 Drake songs."""
    return pd.DataFrame([
        {
            'metadata': {
                'artist': 'Drake',
                'title': 'One Dance',
                'release_date': '2016-04-05',
                'popularity_weight': 95.5
            }
        },
        {
            'metadata': {
                'artist': 'Drake',
                'title': 'Hotline Bling',
                'release_date': '2015-10-19',
                'popularity_weight': 88.2
            }
        },
        {
            'metadata': {
                'artist': 'Drake',
                'title': 'Fake Love',
                'release_date': '2016-10-29',
                'popularity_weight': 82.1
            }
        }
    ])


@pytest.fixture
def empty_df():
    """Empty DataFrame fixture."""
    return pd.DataFrame()


@pytest.fixture
def mock_trends_service():
    """Mock TrendsService for testing."""
    class MockTrendsService:
        def get_brand_trends_from_precomputed(self, **kwargs):
            # Return None to simulate no data
            return None
    return MockTrendsService()


class TestPopularityAnalyzer:
    """Test suite for PopularityAnalyzer class."""

    def test_get_top_songs_returns_correct_count(self, sample_enriched_df):
        """Test that get_top_songs returns correct number of songs."""
        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(sample_enriched_df, 'Nike', top_n=2)

        assert len(songs) == 2
        assert songs[0].title == 'One Dance'
        assert songs[1].title == 'Hotline Bling'

    def test_get_top_songs_sorts_by_popularity(self, sample_enriched_df):
        """Test that songs are sorted by popularity weight (descending)."""
        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(sample_enriched_df, 'Nike', top_n=3)

        assert len(songs) == 3
        assert songs[0].popularity_weight == 95.5
        assert songs[1].popularity_weight == 88.2
        assert songs[2].popularity_weight == 82.1

    def test_get_top_songs_with_all_songs(self, sample_enriched_df):
        """Test requesting more songs than available."""
        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(sample_enriched_df, 'Nike', top_n=10)

        # Should return only 3 (all available)
        assert len(songs) == 3

    def test_empty_dataframe_returns_empty_list(self, empty_df):
        """Test that empty DataFrame returns empty list."""
        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(empty_df, 'Nike')

        assert songs == []
        assert isinstance(songs, list)

    def test_popular_song_model_attributes(self, sample_enriched_df):
        """Test that PopularSong model has correct attributes."""
        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(sample_enriched_df, 'Nike', top_n=1)

        song = songs[0]
        assert isinstance(song, PopularSong)
        assert song.artist == 'Drake'
        assert song.title == 'One Dance'
        assert isinstance(song.release_date, datetime)
        assert song.release_date.year == 2016
        assert song.popularity_weight == 95.5
        assert song.brand == 'Nike'

    def test_get_top_songs_filters_missing_popularity(self):
        """Test that songs without popularity_weight are skipped."""
        df = pd.DataFrame([
            {
                'metadata': {
                    'artist': 'Drake',
                    'title': 'One Dance',
                    'release_date': '2016-04-05',
                    'popularity_weight': 95.5
                }
            },
            {
                'metadata': {
                    'artist': 'Drake',
                    'title': 'No Popularity',
                    'release_date': '2016-04-05'
                    # Missing popularity_weight
                }
            }
        ])

        analyzer = PopularityAnalyzer(None)
        songs = analyzer.get_top_songs(df, 'Nike', top_n=10)

        # Should only return the song with popularity_weight
        assert len(songs) == 1
        assert songs[0].title == 'One Dance'

    def test_find_best_impact_with_no_songs(self, empty_df, mock_trends_service):
        """Test find_best_impact returns None when no songs found."""
        analyzer = PopularityAnalyzer(mock_trends_service)
        result = analyzer.find_best_impact(empty_df, 'Nike')

        assert result is None

    def test_analyzer_initialization(self, mock_trends_service):
        """Test that PopularityAnalyzer initializes correctly."""
        analyzer = PopularityAnalyzer(mock_trends_service)

        assert analyzer.trends_service is mock_trends_service
        assert hasattr(analyzer, 'get_top_songs')
        assert hasattr(analyzer, 'analyze_song_impact')
        assert hasattr(analyzer, 'find_best_impact')
