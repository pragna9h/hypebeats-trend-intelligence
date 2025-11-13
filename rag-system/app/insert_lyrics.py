"""
Insert full song lyrics from lyrics_final.csv into PostgreSQL vector store.
"""
import pandas as pd
import logging
import uuid
import psycopg
from app.config.settings import get_settings
from app.database.vector_store import VectorStore
import os

logger = logging.getLogger(__name__)

def prepare_lyrics_dataframe(csv_path: str) -> pd.DataFrame:
    """Load and prepare lyrics data."""
    logger.info(f"Loading lyrics from {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Filter to only successful cleanings
    df = df[df['status'] == 'Success'].copy()
    logger.info(f"Loaded {len(df)} songs with cleaned lyrics")
    
    contents = []
    metadatas = []
    
    for _, row in df.iterrows():
        # Limit lyrics to 5000 chars for embedding
        lyrics_text = str(row['lyrics_cleaned'])[:5000]
        
        content = f"""Song ID: {row['song_id']}
Lyrics: {lyrics_text}"""
        
        metadata = {
            'song_id': int(row['song_id']),
            'status': str(row['status']),
        }
        
        contents.append(content)
        metadatas.append(metadata)
    
    return pd.DataFrame({
        'id': [str(uuid.uuid4()) for _ in range(len(df))],
        'contents': contents,
        'metadata': metadatas
    })

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(
        script_dir, '..', '..', 'data', 'brand_data', 'lyrics_final.csv'
    ))
    
    logger.info(f"Looking for lyrics at: {csv_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return
    
    try:
        df = prepare_lyrics_dataframe(csv_path)
        
        logger.info("Initializing VectorStore")
        vector_store = VectorStore()
        
        logger.info("Creating full_lyrics table")
        with vector_store.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS full_lyrics (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding vector({vector_store.vector_settings.embedding_dimensions}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            vector_store.conn.commit()
        
        logger.info("Tables created")
        logger.info("Generating embeddings for lyrics (~30 mins, $3.12)...")
        df['embedding'] = df['contents'].apply(vector_store.get_embedding)
        
        logger.info("Inserting records into full_lyrics table")
        with vector_store.conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO full_lyrics (id, metadata, contents, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (row['id'], psycopg.types.json.Jsonb(row['metadata']), 
                      row['contents'], row['embedding']))
            vector_store.conn.commit()
        
        logger.info(f"Inserted {len(df)} records into full_lyrics")
        
        logger.info("Creating IVFFlat index")
        with vector_store.conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS full_lyrics_embedding_idx 
                ON full_lyrics 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            vector_store.conn.commit()
        
        logger.info("✅ Complete! Full lyrics loaded successfully.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()