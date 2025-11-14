# app/query_rag.py
import re
from app.database.vector_store import VectorStore
from app.services.synthesizer import FashionSynthesizer

def extract_date_range(query: str) -> tuple[str, str]:
    """Extract start/end dates from query text."""
    # Pattern for ranges like "2015-2020", "from 2018 to 2023", "between 2015-2020"
    range_patterns = [
        r'(\d{4})\s*-\s*(\d{4})',  # 2015-2020
        r'from\s+(\d{4})\s+to\s+(\d{4})',  # from 2018 to 2023
        r'between\s+(\d{4})\s*-\s*(\d{4})',  # between 2015-2020
        r'from\s+(\d{4})\s+through\s+(\d{4})'  # from 2015 through 2020
    ]
    
    # Try each pattern until we find a match
    for pattern in range_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            start, end = match.groups()
            return f"{start}-01-01", f"{end}-12-31"
    
    # Pattern for single year queries
    single_year = re.search(r'(?:in|from|year|of)\s+(\d{4})\b', query, re.IGNORECASE)
    if single_year:
        year = single_year.group(1)
        return f"{year}-01-01", f"{year}-12-31"
    
    return None, None

def query_system(question: str, limit: int = 50):
    """Complete RAG pipeline with enriched data and temporal filtering."""
    print(f"\n{'='*80}\nQ: {question}\n{'='*80}")
    
    # Extract date range from query
    start_date, end_date = extract_date_range(question)
    print(f"📅 Extracted dates - Start: {start_date}, End: {end_date}")
    
    if start_date and end_date:
        print(f"📅 Filtering: {start_date} to {end_date}")
    
    # Retrieve with JOINs and date filtering
    vs = VectorStore()
    try:
        results = vs.search_with_joins(
            question, 
            limit=limit, 
            start_date=start_date,
            end_date=end_date,
            return_dataframe=True
        )
        print(f"✓ Retrieved {len(results)} mentions with enriched context")

    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
        raise
    
    # Generate
    synth = FashionSynthesizer()
    insight = synth.generate_insight(question, results)
    
    # Display
    print(f"\n{insight.summary}\n")
    for finding in insight.key_findings:
        print(f"• {finding}")
    print(f"\nQuality: {insight.data_quality}\n")

    # Display sample data
    if not results.empty:
        unique_artists = results['artist_name'].unique()
        print(f"Artists: {', '.join([str(a) for a in unique_artists if a])}\n")
        
        recent = results.sort_values('release_date', ascending=False).head(10)
        print("Sample mentions:")
        print(recent[['artist_name', 'song_title', 'release_date']].to_string(index=False))
    
    vs.close()

if __name__ == "__main__":
    queries = [
        "Which year did Gucci have the most mentions in songs?",
        "What was the most mentioned brand in 2022?",
        "What brands does Travis Scott mention most?",
        "What brands does Kanye West mention most?",
        "What brands does Future mention most?",
        "What clothing item does Kanye West mention most?",
        "How has the popularity of boots changed over time?",
    ]
        
    for q in queries:
        query_system(q)