"""
Diagnostic tool to inspect enriched_lyrics DataFrame structure
"""
import pandas as pd
import json
from sqlalchemy import create_engine, text
from app.config.settings import get_settings

def inspect_enriched_structure():
    """Print actual structure of enriched_lyrics data"""
    settings = get_settings()
    engine = create_engine(settings.database.service_url)
    
    # Get sample rows for Future
    query = """
    SELECT * FROM enriched_lyrics 
    WHERE metadata->>'artist' = 'Future'
    LIMIT 3
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        print("=" * 80)
        print("DATAFRAME COLUMNS:")
        print(df.columns.tolist())
        print("\n" + "=" * 80)
        
        if not df.empty:
            print("SAMPLE ROW (row 0):")
            for col in df.columns:
                value = df.iloc[0][col]
                print(f"  {col}: {value}")
            
            print("\n" + "=" * 80)
            print("METADATA STRUCTURE:")
            if 'metadata' in df.columns:
                meta = df.iloc[0]['metadata']
                print(f"  Raw type: {type(meta)}")
                print(f"  Raw value: {meta}")
                
                # Parse if it's a string
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                        print(f"  Parsed type: {type(meta)}")
                    except:
                        print("  Failed to parse as JSON")
                        meta = {}
                
                if isinstance(meta, dict):
                    print(f"  Dict keys: {list(meta.keys())}")
                    print("  Dict contents:")
                    for key, value in meta.items():
                        print(f"    {key}: {value}")
                else:
                    print("  Metadata is not a dict")
            else:
                print("  NO 'metadata' COLUMN FOUND")
        else:
            print("  NO DATA FOUND for Future")
            
        # Also check what artists are available
        print("\n" + "=" * 80)
        print("AVAILABLE ARTISTS (sample):")
        artist_query = """
        SELECT DISTINCT metadata->>'artist' as artist 
        FROM enriched_lyrics 
        WHERE metadata->>'artist' IS NOT NULL
        LIMIT 10
        """
        artists_df = pd.read_sql(artist_query, engine)
        for i, row in artists_df.iterrows():
            print(f"  {row['artist']}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    inspect_enriched_structure()
