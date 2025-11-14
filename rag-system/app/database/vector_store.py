import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
import pandas as pd
from openai import OpenAI
from app.config.settings import get_settings

class VectorStore:
    """Manages vector operations and database interactions."""

    def __init__(self):
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.conn = psycopg.connect(self.settings.database.service_url)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()
        register_vector(self.conn)

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f}s")
        return embedding

    def create_tables(self) -> None:
        """Create necessary tables in database."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.vector_settings.table_name} (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding vector({self.vector_settings.embedding_dimensions}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
        logging.info("Tables created")

    def create_index(self) -> None:
        """Create IVFFlat index for faster similarity search."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.vector_settings.table_name}_embedding_idx 
                ON {self.vector_settings.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            self.conn.commit()
        logging.info("Index created")

    def drop_index(self) -> None:
        """Drop index."""
        with self.conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {self.vector_settings.table_name}_embedding_idx;")
            self.conn.commit()

    def upsert(self, df: pd.DataFrame) -> None:
        """Insert or update records from DataFrame."""
        with self.conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(f"""
                    INSERT INTO {self.vector_settings.table_name} (id, metadata, contents, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE 
                    SET metadata = EXCLUDED.metadata,
                        contents = EXCLUDED.contents,
                        embedding = EXCLUDED.embedding;
                """, (row['id'], psycopg.types.json.Jsonb(row['metadata']), row['contents'], row['embedding']))
            self.conn.commit()
        logging.info(f"Inserted {len(df)} records into {self.vector_settings.table_name}")

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        return_dataframe: bool = True,
        table_name: str = "embeddings"
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Query vector database for similar embeddings."""
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()

        with self.conn.cursor(row_factory=dict_row) as cur:
            sql = f"""
                SELECT id, metadata, contents, embedding,
                       1 - (embedding <=> %s::vector) as distance
                FROM {table_name}
            """
            params = [query_embedding]
            
            if metadata_filter:
                conditions = []
                for key, value in metadata_filter.items():
                    conditions.append(f"metadata->>%s = %s")
                    params.extend([key, str(value)])
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += f" ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, limit])
            
            cur.execute(sql, params)
            results = cur.fetchall()

        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.3f}s")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        return results

    def search_with_joins(
        self,
        query_text: str,
        limit: int = 200,
        start_date: str = None,
        end_date: str = None,
        return_dataframe: bool = True
    ) -> Union[List[dict], pd.DataFrame]:
        """Search both brand_mentions and full_lyrics with JOINs and date filtering."""
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()

        with self.conn.cursor(row_factory=dict_row) as cur:
            # Build date filter clause
            date_filter = ""
            date_params = []
            
            if start_date or end_date:
                date_conditions = []
                
                if start_date:
                    date_conditions.append("""
                        (
                            (s.release_date ~ '^\\d{1,2}/\\d{1,2}/\\d{4}$' AND TO_DATE(s.release_date, 'FMMM/FMDD/YYYY') >= %s::date) OR
                            (s.release_date ~ '^\\d{4}$' AND s.release_date::int >= EXTRACT(YEAR FROM %s::date)::int)
                        )
                    """)
                    date_params.extend([start_date, start_date])
                
                if end_date:
                    date_conditions.append("""
                        (
                            (s.release_date ~ '^\\d{1,2}/\\d{1,2}/\\d{4}$' AND TO_DATE(s.release_date, 'FMMM/FMDD/YYYY') <= %s::date) OR
                            (s.release_date ~ '^\\d{4}$' AND s.release_date::int <= EXTRACT(YEAR FROM %s::date)::int)
                        )
                    """)
                    date_params.extend([end_date, end_date])
                
                date_filter = " AND " + " AND ".join(date_conditions)
            
            sql = f"""
                (
                    SELECT 
                        bm.id::text as id,
                        bm.contents,
                        'brand_mention' as source,
                        s.song_title,
                        s.release_date,
                        a.artist_name,
                        a.genre,
                        a.region,
                        b.brand_name,
                        b.category,
                        1 - (bm.embedding <=> %s::vector) as similarity
                    FROM brand_mentions bm
                    LEFT JOIN songs s ON bm.song_id = s.song_id
                    LEFT JOIN artists a ON s.artist_id = a.artist_id
                    LEFT JOIN brands b ON bm.brand_id = b.brand_id
                    WHERE 1=1 {date_filter}
                )
                UNION ALL
                (
                    SELECT 
                        fl.song_id::text as id,
                        SUBSTRING(fl.contents FROM 1 FOR 500) as contents,
                        'full_lyrics' as source,
                        s.song_title,
                        s.release_date,
                        a.artist_name,
                        a.genre,
                        a.region,
                        NULL as brand_name,
                        NULL as category,
                        1 - (fl.embedding <=> %s::vector) as similarity
                    FROM full_lyrics fl
                    LEFT JOIN songs s ON fl.song_id = s.song_id
                    LEFT JOIN artists a ON s.artist_id = a.artist_id
                    WHERE 1=1 {date_filter}
                )
                ORDER BY similarity DESC
                LIMIT %s
            """
            
            # Build params: embedding for each subquery + date params (2x) + limit
            params = [query_embedding] + date_params + [query_embedding] + date_params + [limit]
            
            cur.execute(sql, params)
            results = cur.fetchall()

        elapsed_time = time.time() - start_time
        logging.info(f"Combined search completed in {elapsed_time:.3f}s, found {len(results)} results")

        if return_dataframe:
            return pd.DataFrame(results)
        return results

    def _create_dataframe_from_results(self, results: List[dict]) -> pd.DataFrame:
        """Convert search results to DataFrame."""
        df = pd.DataFrame(results)
        if not df.empty:
            metadata_df = pd.json_normalize(df['metadata'])
            df = pd.concat([df.drop(['metadata'], axis=1), metadata_df], axis=1)
            df['id'] = df['id'].astype(str)
        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records."""
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError("Provide exactly one of: ids, metadata_filter, or delete_all")

        with self.conn.cursor() as cur:
            if delete_all:
                cur.execute(f"DELETE FROM {self.vector_settings.table_name};")
            elif ids:
                cur.execute(f"DELETE FROM {self.vector_settings.table_name} WHERE id = ANY(%s);", (ids,))
            elif metadata_filter:
                conditions = [f"metadata->>%s = %s" for key in metadata_filter.keys()]
                cur.execute(
                    f"DELETE FROM {self.vector_settings.table_name} WHERE {' AND '.join(conditions)};",
                    [item for pair in metadata_filter.items() for item in pair]
                )
            self.conn.commit()
        logging.info("Deletion completed")

    def close(self):
        self.conn.close()