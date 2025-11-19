# app/query_rag.py
import re
from datetime import datetime, timedelta
from collections import defaultdict
from app.database.vector_store import VectorStore
from app.services.synthesizer import FashionSynthesizer
from app.services.trends_service import TrendsService, TrendDecision
from app.models.trends import TrendsRequest
import instructor
from openai import OpenAI

def extract_date_range(query: str) -> tuple[str, str]:
    """Extract start/end dates from query text."""
    range_patterns = [
        r'(\d{4})\s*-\s*(\d{4})',
        r'from\s+(\d{4})\s+to\s+(\d{4})',
        r'between\s+(\d{4})\s*-\s*(\d{4})',
        r'from\s+(\d{4})\s+through\s+(\d{4})'
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            start, end = match.groups()
            return f"{start}-01-01", f"{end}-12-31"
    
    single_year = re.search(r'(?:in|from|year|of)\s+(\d{4})\b', query, re.IGNORECASE)
    if single_year:
        year = single_year.group(1)
        return f"{year}-01-01", f"{year}-12-31"
    
    return None, None

def parse_release_date(date_str: str) -> datetime | None:
    """Parse various date formats from database to datetime."""
    if not date_str or date_str == 'Unknown':
        return None
    
    try:
        return datetime.strptime(str(date_str), '%m/%d/%Y')
    except:
        pass
    
    try:
        return datetime.strptime(str(date_str), '%Y-%m-%d')
    except:
        pass
    
    try:
        return datetime.strptime(str(date_str), '%Y')
    except:
        pass
    
    return None

def cluster_mentions_by_month(results):
    """Group mentions by month, return {month: [dates]}."""
    monthly_clusters = defaultdict(list)
    
    for date_str in results['release_date'].dropna().unique():
        parsed = parse_release_date(date_str)
        if parsed:
            month_key = parsed.strftime('%Y-%m')
            monthly_clusters[month_key].append(parsed)
    
    # Sort dates within each month
    for month in monthly_clusters:
        monthly_clusters[month] = sorted(monthly_clusters[month])
    
    return dict(monthly_clusters)

def query_system(question: str, limit: int = 50):
    """Complete RAG pipeline with monthly clustering."""
    print(f"\n{'='*80}\nQ: {question}\n{'='*80}")
    
    start_date, end_date = extract_date_range(question)
    print(f"📅 Extracted dates - Start: {start_date}, End: {end_date}")
    
    if start_date and end_date:
        print(f"📅 Filtering: {start_date} to {end_date}")
    
    # Check if query needs trends data
    client = instructor.from_openai(OpenAI())
    decision = client.chat.completions.create(
        model="gpt-4o",
        response_model=TrendDecision,
        messages=[{
            "role": "system",
            "content": "Determine if query needs Google Trends data and extract brand/dates."
        }, {
            "role": "user",
            "content": question
        }]
    )
    
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
    
    # Cluster mentions by month
    monthly_clusters = cluster_mentions_by_month(results)
    print(f"📊 Clustered into {len(monthly_clusters)} months")
    
    # Fetch trends data if needed
    trends_data_list = []
    if decision.needs_trends and decision.brand:
        print(f"📈 Fetching trends for {decision.brand}...")
        trends_service = TrendsService()
        
        # Filter clusters with 3+ mentions
        significant_months = {
            month: dates for month, dates in monthly_clusters.items() 
            if len(dates) >= 3
        }
        
        print(f"🎯 Found {len(significant_months)} months with 3+ mentions")
        
        for month, dates in sorted(significant_months.items()):
            try:
                # Calculate window: 1 month before to 2 months after
                first_date = dates[0]
                window_start = (first_date - timedelta(days=30)).strftime('%Y-%m-%d')
                window_end = (first_date + timedelta(days=60)).strftime('%Y-%m-%d')
                
                print(f"  📅 {month}: {len(dates)} mentions, window: {window_start} to {window_end}")
                
                trends_data = trends_service.get_brand_trends(
                    request=TrendsRequest(
                        brand=decision.brand,
                        start_date=window_start,
                        end_date=window_end
                    ),
                    mention_dates=[d.isoformat() for d in dates]
                )
                
                trends_data_list.append({
                    'month': month,
                    'mention_count': len(dates),
                    'trends': trends_data
                })
                
                if trends_data.pre_mention_avg > 0:
                    print(f"     Pre: {trends_data.pre_mention_avg}, Post: {trends_data.post_mention_avg}, Change: {trends_data.percent_change:+.1f}%")
                    
            except Exception as e:
                print(f"  ⚠️  {month} trends failed: {e}")
    
    # Generate
    synth = FashionSynthesizer()
    insight = synth.generate_insight(question, results, trends_data_list, monthly_clusters)
    
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
        "Which 10 brands had the fastest growth after mentions in 2022?",  # Strong effects
        "Which brands showed no growth after mentions in 2022?",           # Null effects
        "Did Travis Scott's mentions affect brand popularity in 2022?",   # Artist influence
        "What clothing items are most mentioned with Balenciaga?",         # Item-level
        "How did Dior's popularity change across 2020-2022?",             # Multi-year trends
        "Are luxury brand mentions increasing from 2015-2025?",
    ]
            
    # Test all queries
    for q in queries:
        query_system(q)