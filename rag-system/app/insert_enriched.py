import json
import logging
import pandas as pd
from uuid import uuid4
from app.config.settings import get_settings
from app.database.vector_store import VectorStore

settings = get_settings()
logger = logging.getLogger(__name__)

def prepare_dataframe(jsonl_path: str) -> pd.DataFrame:
    """Load JSONL and convert to DataFrame format for VectorStore."""
    logger.info(f"Loading data from {jsonl_path}")
    
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    logger.info(f"Loaded {len(records)} records")
    
    # Prepare data in VectorStore format
    data = []
    for r in records:
        # Content for embedding
        content = f"""Artist: {r['artist']}
Song: {r['title']}
Released: {r['release_date']} ({r['release_quarter']})
Clothing: {r['canonical_label']} (as "{r['surface_form']}")
Type: {r['mention_type']}
Context: {r['context_window']}
Popularity: {r.get('popularity_weight', 0):.2f}"""
        
        # Metadata
        metadata = {
            "song_id": r.get('song_id'),
            "artist": r.get('artist'),
            "title": r.get('title'),
            "release_date": r.get('release_date'),
            "release_quarter": r.get('release_quarter'),
            "mention_type": r.get('mention_type'),
            "surface_form": r.get('surface_form'),
            "canonical_label": r.get('canonical_label'),
            "pageviews": r.get('pageviews'),
            "popularity_weight": r.get('popularity_weight'),
        }
        
        data.append({
            "id": str(uuid4()),
            "metadata": metadata,
            "contents": content,
        })
    
    return pd.DataFrame(data)

def main():
    import os
    # Path to data file in the project's data directory
    jsonl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "data", "lyrics_mentions_enriched_v2.jsonl")
    
    # Prepare DataFrame
    df = prepare_dataframe(jsonl_path)
    
    # Initialize VectorStore
    logger.info("Initializing VectorStore")
    vector_store = VectorStore()
    
    # Create tables
    logger.info("Creating tables")
    vector_store.create_tables()
    
    # Generate embeddings and upsert
    logger.info("Generating embeddings and inserting records")
    df['embedding'] = df['contents'].apply(vector_store.get_embedding)
    vector_store.upsert(df)
    
    # Create index for fast similarity search
    logger.info("Creating StreamingDiskANN index")
    vector_store.create_index()
    
    logger.info("✅ Complete!")

if __name__ == "__main__":
    main()