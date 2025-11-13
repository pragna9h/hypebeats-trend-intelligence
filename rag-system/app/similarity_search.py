import logging
from app.config.settings import get_settings
from app.database.vector_store import VectorStore

settings = get_settings()
logger = logging.getLogger(__name__)

def main():
    vector_store = VectorStore()
    
    queries = [
        ("A$AP Rocky mentions Rick Owens", {"artist": "A$AP Rocky"}),
        ("Kanye West talks about Yeezy sneakers", {"artist": "Kanye West"}),
        ("luxury fashion brands in hip-hop 2024", {}),
        ("How many times does Travis Scott say white tee in Franchise", {"artist": "Travis Scott"}),
    ]
    
    for query, metadata_filter in queries:
        print(f"\n{'#'*80}\nQuery: {query}\n{'#'*80}")
        
        # Use VectorStore search method
        results = vector_store.search(
            query_text=query,
            limit=3,
            metadata_filter=metadata_filter if metadata_filter else None,
            return_dataframe=True
        )
        
        print(f"\nFound {len(results)} results:\n")
        for i, row in results.iterrows():
            print(f"Result {i+1} (Distance: {row['distance']:.4f})")
            print(f"  {row['artist']} - {row['title']} ({row['release_date']})")
            print(f"  {row['canonical_label']} (as '{row['surface_form']}')")
            print(f"  Context: {row.get('contents', 'N/A')}...\n")

if __name__ == "__main__":
    main()