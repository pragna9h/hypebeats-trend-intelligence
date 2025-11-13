"""
Insert brand mentions from mentions.csv into PostgreSQL vector store.
Joins with songs, artists, and brands tables for complete metadata.
"""
import pandas as pd
import logging
import uuid
import psycopg
from app.config.settings import get_settings
from app.database.vector_store import VectorStore
import os

logger = logging.getLogger(__name__)

def prepare_mentions_dataframe(data_dir: str) -> pd.DataFrame:
    """Load and prepare the mentions data with related metadata."""
    logger.info(f"Loading data from {data_dir}")
    
    # Load all CSV files
    try:
        mentions_df = pd.read_csv(os.path.join(data_dir, 'mentions.csv'), encoding='utf-8-sig')
        songs_df = pd.read_csv(os.path.join(data_dir, 'songs_final.csv'), encoding='utf-8-sig')
        brands_df = pd.read_csv(os.path.join(data_dir, 'brands.csv'), encoding='utf-8-sig')
        artists_df = pd.read_csv(os.path.join(data_dir, 'artists.csv'), encoding='utf-8-sig')
    except FileNotFoundError as e:
        logger.error(f"Error loading CSV files: {str(e)}")
        raise

    # Merge data with proper suffixes
    logger.info(f"Loaded {len(mentions_df)} mentions")
    df = mentions_df.merge(songs_df, on='song_id', how='left')
    df = df.merge(brands_df, on='brand_id', how='left', suffixes=('_mention', '_brand'))
    df = df.merge(artists_df, on='artist_id', how='left')

    # Prepare content and metadata
    contents = []
    metadatas = []
    
    for _, row in df.iterrows():
        # Get brand alias with fallback
        brand_alias = row.get('brand_alias_mention', row.get('brand_alias', ''))
        
        # Format content
        content = f"""Artist: {row['artist_name']}
Song: {row['song_title']}
Released: {row['release_date']}
Brand: {row['brand_name']} (artist said "{brand_alias}")
Category: {row.get('category', '')}
Context: {str(row['context'])[:300]}"""
        
        # Prepare metadata
        metadata = {
            'mention_id': int(row['mention_id']),
            'song_id': int(row['song_id']),
            'brand_id': int(row['brand_id']),
            'artist_id': int(row['artist_id']),
            'artist_name': str(row['artist_name']),
            'song_title': str(row['song_title']),
            'brand_name': str(row['brand_name']),
            'release_date': str(row['release_date']),
            'category': str(row.get('category', '')),
        }
        
        contents.append(content)
        metadatas.append(metadata)
    
    return pd.DataFrame({
        'id': [str(uuid.uuid4()) for _ in range(len(df))],
        'contents': contents,
        'metadata': metadatas
    })

def main():
    # Path to brand_data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(
        script_dir, '..', '..', 'data', 'brand_data'
    ))
    
    logger.info(f"Looking for data in: {data_dir}")
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        print(f"Error: Data directory not found at: {data_dir}")
        return
    
    try:
        df = prepare_mentions_dataframe(data_dir)
        
        logger.info("Initializing VectorStore")
        vector_store = VectorStore()
        
        logger.info("Creating brand_mentions table")
        with vector_store.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS brand_mentions (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding vector({vector_store.vector_settings.embedding_dimensions}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            vector_store.conn.commit()
        
        logger.info("Tables created")
    
        logger.info("Generating embeddings for brand mentions...")
        df['embedding'] = df['contents'].apply(vector_store.get_embedding)
        
        logger.info("Inserting records into brand_mentions table")
        with vector_store.conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO brand_mentions (id, metadata, contents, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (row['id'], psycopg.types.json.Jsonb(row['metadata']), 
                      row['contents'], row['embedding']))
            vector_store.conn.commit()
        
        logger.info(f"Inserted {len(df)} records into brand_mentions")
        
        logger.info("Creating IVFFlat index")
        with vector_store.conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS brand_mentions_embedding_idx 
                ON brand_mentions 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            vector_store.conn.commit()
        
        logger.info("✅ Complete! Brand mentions loaded successfully.")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()