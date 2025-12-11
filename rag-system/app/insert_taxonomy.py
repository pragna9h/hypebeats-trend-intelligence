"""
Insert taxonomy trends from taxonomy_trends_detailed.csv into PostgreSQL vector store.
Aggregates monthly data per item, embeds item descriptions only.
"""
import pandas as pd
import logging
import uuid
import psycopg
from app.config.settings import get_settings
from app.database.vector_store import VectorStore
import os

logger = logging.getLogger(__name__)

def prepare_taxonomy_dataframe(csv_path: str) -> pd.DataFrame:
    """Load and prepare taxonomy data with monthly trends as metadata."""
    logger.info(f"Loading taxonomy from {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    logger.info(f"Loaded {len(df)} rows")
    
    # Group by canonical_label to aggregate monthly trends
    grouped = df.groupby('canonical_label')
    logger.info(f"Found {len(grouped)} unique items")
    
    contents = []
    metadatas = []
    
    for label, group in grouped:
        # Sort by month
        group = group.sort_values('month')
        
        # Content for embedding (semantic search)
        content = f"{group.iloc[0]['canonical_label']} - {group.iloc[0]['category']} {group.iloc[0]['label_type']}"
        
        # Monthly trends as list
        monthly_trends = [
            {"month": row['month'], "interest": float(row['interest'])}
            for _, row in group.iterrows()
        ]
        
        # Calculate stats
        interests = [t['interest'] for t in monthly_trends]
        recent_interests = [t['interest'] for t in monthly_trends[-3:]]  # Last 3 months
        
        metadata = {
            'canonical_label': str(group.iloc[0]['canonical_label']),
            'label_type': str(group.iloc[0]['label_type']),
            'category': str(group.iloc[0]['category']),
            'monthly_trends': monthly_trends,
            'stats': {
                'peak': max(interests),
                'avg': sum(interests) / len(interests),
                'recent_3mo_avg': sum(recent_interests) / len(recent_interests) if recent_interests else 0
            }
        }
        
        contents.append(content)
        metadatas.append(metadata)
    
    return pd.DataFrame({
        'id': [str(uuid.uuid4()) for _ in range(len(grouped))],
        'contents': contents,
        'metadata': metadatas
    })

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(
        script_dir, '..', 'data', 'taxonomy_trends_detailed.csv'
    ))
    
    logger.info(f"Looking for taxonomy at: {csv_path}")
    
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return
    
    try:
        df = prepare_taxonomy_dataframe(csv_path)
        
        logger.info("Initializing VectorStore")
        vector_store = VectorStore()
        
        logger.info("Creating taxonomy_items table")
        with vector_store.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS taxonomy_items (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding vector({vector_store.vector_settings.embedding_dimensions}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            vector_store.conn.commit()
        
        logger.info("Generating embeddings for 60 items (~5 secs, $0.001)...")
        df['embedding'] = df['contents'].apply(vector_store.get_embedding)
        
        logger.info("Inserting records into taxonomy_items table")
        with vector_store.conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO taxonomy_items (id, metadata, contents, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (row['id'], psycopg.types.json.Jsonb(row['metadata']), 
                      row['contents'], row['embedding']))
            vector_store.conn.commit()
        
        logger.info(f"Inserted {len(df)} items into taxonomy_items")
        
        logger.info("Creating IVFFlat index")
        with vector_store.conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS taxonomy_items_embedding_idx 
                ON taxonomy_items 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10);
            """)
            vector_store.conn.commit()
        
        logger.info("✅ Complete! Taxonomy loaded successfully.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()