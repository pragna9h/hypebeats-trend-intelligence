# app/load_brand_trends.py
import pandas as pd
from sqlalchemy import create_engine, text
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

    # Path to data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, 'brand_trends_monthly.csv')
    logger.info(f"Loading brand trends from: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Convert date column
    df['month'] = pd.to_datetime(df['month'])

    # Ensure numeric column is correct type
    df['interest'] = pd.to_numeric(df['interest'], errors='coerce')

    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['label', 'month'], keep='first')
    logger.info(f"After deduplication: {len(df)} rows")

    # Create table with proper schema and indexes
    with engine.connect() as conn:
        # Drop existing table if exists
        conn.execute(text("DROP TABLE IF EXISTS brand_trends_monthly"))

        # Create table
        conn.execute(text("""
            CREATE TABLE brand_trends_monthly (
                id SERIAL,
                label VARCHAR(255) NOT NULL,
                query VARCHAR(255),
                month DATE NOT NULL,
                interest FLOAT,
                PRIMARY KEY (label, month)
            )
        """))

        # Create index for faster lookups
        conn.execute(text("""
            CREATE INDEX idx_brand_trends_label_month
            ON brand_trends_monthly (LOWER(label), month)
        """))

        conn.commit()
        logger.info("✓ Created table brand_trends_monthly with indexes")

    # Load data
    df.to_sql('brand_trends_monthly', engine, if_exists='append', index=False)
    logger.info(f"✓ Loaded {len(df)} rows into brand_trends_monthly")

    # Verify data
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) as count FROM brand_trends_monthly"))
        count = result.fetchone()[0]
        logger.info(f"✓ Verified {count} total rows in table")

        # Show sample brands
        result = conn.execute(text("""
            SELECT label, COUNT(*) as months
            FROM brand_trends_monthly
            GROUP BY label
            ORDER BY months DESC
            LIMIT 5
        """))
        logger.info("Top 5 brands by month count:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]} months")

    logger.info("✅ Complete")

if __name__ == "__main__":
    main()