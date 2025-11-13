# app/services/load_metadata_tables.py
import pandas as pd
from sqlalchemy import create_engine
from app.config.settings import get_settings
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    engine = create_engine(settings.database.service_url)
    
    # Go up three levels from current file to reach the project root
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data', 'brand_data'
    ))
    logger.info(f"Loading data from: {base_path}")
    
    # Load in order (FK constraints)
    for file, table in [
        ('artists.csv', 'artists'),
        ('brands.csv', 'brands'),
        ('songs_final.csv', 'songs')
    ]:
        df = pd.read_csv(os.path.join(base_path, file), encoding='utf-8-sig')
        df.to_sql(table, engine, if_exists='replace', index=False)
        logger.info(f"✓ Loaded {len(df)} rows into {table}")
    
    logger.info("✅ Complete")

if __name__ == "__main__":
    main()