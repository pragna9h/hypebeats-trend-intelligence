# app/query_rag.py
from app.database.vector_store import VectorStore
from app.services.synthesizer import FashionSynthesizer

def query_system(question: str, limit: int = 5):
    """Complete RAG pipeline with enriched data."""
    print(f"\n{'='*80}\nQ: {question}\n{'='*80}")
    
    # Retrieve with JOINs
    vs = VectorStore()
    results = vs.search_with_joins(question, limit=limit, return_dataframe=True)
    print(f"✓ Retrieved {len(results)} mentions with enriched context")
    
    # Generate
    synth = FashionSynthesizer()
    insight = synth.generate_insight(question, results)
    
    # Display
    print(f"\n{insight.summary}\n")
    for finding in insight.key_findings:
        print(f"• {finding}")
    print(f"\nQuality: {insight.data_quality}\n")
    
    vs.close()

if __name__ == "__main__":
    queries = [
        "What fashion items does Lil Baby mention most?",
        "Show me expensive jewelry and luxury accessories in hip-hop",
        "Compare fashion styles: Lil Baby vs Future",
        "What luxury brands does Future mention?",
    ]
    
    for q in queries:
        query_system(q)