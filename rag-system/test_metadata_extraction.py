"""Test that metadata extraction works correctly"""
import pandas as pd
from app.services.popularity_analyzer import PopularityAnalyzer

def test_metadata_extraction():
    """Verify artist/title extraction from enriched_lyrics"""
    
    # Create test DataFrame matching actual structure
    test_data = pd.DataFrame([
        {
            'id': 1,
            'contents': 'Future wears a ring in his new song',
            'metadata': {  # Nested dict - this is the actual structure
                'artist': 'Future',
                'title': 'WAIT FOR U',
                'release_date': '2022-04-29',
                'popularity_weight': 15.4,
                'song_id': 12345,
                'pageviews': 1000000
            },
            'embedding': [],
            'created_at': '2024-01-01'
        },
        {
            'id': 2,
            'contents': 'Future with more accessories',
            'metadata': {
                'artist': 'Future',
                'title': 'Mask Off',
                'release_date': '2017-02-16',
                'popularity_weight': 14.2,
                'song_id': 67890,
                'pageviews': 800000
            },
            'embedding': [],
            'created_at': '2024-01-01'
        }
    ])
    
    print("Testing metadata extraction...")
    print(f"Test DataFrame columns: {test_data.columns.tolist()}")
    print(f"Test metadata structure: {test_data.iloc[0]['metadata']}")
    
    analyzer = PopularityAnalyzer(None)
    songs = analyzer.get_top_songs(test_data, 'Nike', top_n=3)
    
    print(f"\nResults: Found {len(songs)} songs")
    
    assert len(songs) == 2, f"Expected 2 songs, got {len(songs)}"
    
    for i, song in enumerate(songs, 1):
        print(f"  {i}. '{song.title}' by {song.artist} (pop: {song.popularity_weight})")
        assert song.artist != 'Unknown', f"Artist extraction failed: {song.artist}"
        assert song.title != 'Unknown', f"Title extraction failed: {song.title}"
    
    # Check popularity ranking
    assert songs[0].popularity_weight == 15.4, f"Wrong ranking: {songs[0].popularity_weight}"
    assert songs[1].popularity_weight == 14.2, f"Wrong ranking: {songs[1].popularity_weight}"
    
    print("\n✅ Metadata extraction test PASSED")
    print(f"   Top song: {songs[0].title} by {songs[0].artist}")
    print(f"   Popularity weights: {[s.popularity_weight for s in songs]}")

if __name__ == "__main__":
    test_metadata_extraction()
