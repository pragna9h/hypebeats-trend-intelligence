"""
Test taxonomy search functionality.
"""
import sys
sys.path.append('/Users/aaditya/Desktop/HYPEBEATS_GH/Untitled/hypebeats/rag-system')  # Updated path

from app.database.vector_store import VectorStore
import json

def test_taxonomy_search():
    vs = VectorStore()
    
    test_queries = [
        "sneakers",
        "Air Force 1s",
        "puffer jacket",
        "luxury handbags",
        "athletic shoes"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        results = vs.search_taxonomy(query, limit=3, return_dataframe=True)
        
        if results.empty:
            print("No results found")
            continue
        
        for idx, row in results.iterrows():
            print(f"\n{idx+1}. {row['canonical_label']}")
            print(f"   Category: {row['category']} | Type: {row['label_type']}")
            print(f"   Similarity: {row['similarity']:.3f}")
            
            # Parse stats
            stats = json.loads(row['stats']) if isinstance(row['stats'], str) else row['stats']
            print(f"   Peak interest: {stats['peak']}")
            print(f"   Avg interest: {stats['avg']:.2f}")
            print(f"   Recent 3mo avg: {stats['recent_3mo_avg']:.2f}")
            
            # Show sample trend data
            trends = json.loads(row['monthly_trends']) if isinstance(row['monthly_trends'], str) else row['monthly_trends']
            print(f"   Total months: {len(trends)}")
            print(f"   First 3 months:")
            for t in trends[:3]:
                print(f"     {t['month']}: {t['interest']}")
    
    vs.close()

if __name__ == "__main__":
    test_taxonomy_search()
