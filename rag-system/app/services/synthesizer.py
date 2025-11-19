"""
Synthesizer: Generates answers from retrieved context using structured outputs.
"""
from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from app.services.llm_factory import LLMFactory

class FashionInsight(BaseModel):
    """Structured response for fashion analytics queries."""
    summary: str = Field(description="2-3 sentence answer to the question")
    key_findings: List[str] = Field(description="Bullet points of specific mentions found")
    data_quality: str = Field(description="'sufficient', 'partial', or 'insufficient'")

class FashionSynthesizer:
    """Generates insights from lyric mention data."""
    
    SYSTEM_PROMPT = """You are a fashion analytics AI analyzing hip-hop lyric brand mentions.

Rules:
- Use ONLY the provided data
- Cite specific examples (artist, song, date, brand)
- If multiple artists mention same brand, note the pattern
- Be quantitative: "Drake mentioned Nike 3 times"
- When monthly clusters are provided, analyze cumulative momentum effects
- Look for patterns: does a cluster of mentions in one month correlate with trend spikes 2-4 weeks later?
- Consider lag effects and cultural momentum building over time
- Mark data_quality as 'insufficient' if can't answer properly"""

    def __init__(self):
        self.llm = LLMFactory("openai")
    
    def generate_insight(
        self, 
        question: str, 
        context_df: pd.DataFrame, 
        trends_data_list: list = None, 
        monthly_clusters: dict = None
    ) -> FashionInsight:
        """Generate structured insight from retrieved mentions and monthly trends."""
        context_str = self._format_context(context_df)
        
        # Add monthly cluster analysis
        trends_str = ""
        if trends_data_list and monthly_clusters:
            trends_str = f"\n\nMonthly Cluster Analysis:\n"
            trends_str += f"Total clusters: {len(monthly_clusters)} months\n"
            trends_str += f"Significant clusters (3+ mentions): {len(trends_data_list)}\n\n"
            
            for cluster_data in trends_data_list:
                month = cluster_data['month']
                count = cluster_data['mention_count']
                trends = cluster_data['trends']
                
                trends_str += f"📊 {month} Cluster ({count} mentions):\n"
                trends_str += f"   Brand: {trends.brand}\n"
                trends_str += f"   Window: {trends.timeframe}\n"
                trends_str += f"   Avg interest: {trends.average_interest}\n"
                
                if trends.pre_mention_avg > 0:
                    trends_str += f"   Pre-cluster avg: {trends.pre_mention_avg}\n"
                    trends_str += f"   Post-cluster avg: {trends.post_mention_avg}\n"
                    trends_str += f"   Change: {trends.percent_change:+.1f}%\n"
                
                # Show temporal trend data
                if trends.data:
                    trends_str += f"   Trend timeline ({len(trends.data)} weeks):\n"
                    for point in trends.data[:3]:
                        trends_str += f"     {point.date.strftime('%Y-%m-%d')}: {point.value}\n"
                    if len(trends.data) > 6:
                        trends_str += f"     ...\n"
                        for point in trends.data[-3:]:
                            trends_str += f"     {point.date.strftime('%Y-%m-%d')}: {point.value}\n"
                
                trends_str += "\n"
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nData:\n{context_str}{trends_str}"}
        ]
        
        return self.llm.create_completion(
            response_model=FashionInsight,
            messages=messages,
            model="gpt-5"
        )
    
    def _format_context(self, df: pd.DataFrame) -> str:
        """Format DataFrame with enriched fields."""
        if df.empty:
            return "No mentions found."
        
        lines = []
        for idx, row in df.iterrows():
            artist = row.get('artist_name', 'Unknown')
            title = row.get('song_title', 'Unknown')
            date = row.get('release_date', 'Unknown')
            brand = row.get('brand_name', 'Unknown')
            genre = row.get('genre', '')
            category = row.get('category', '')
            
            content = row.get('contents', '')
            lyric = content[:100] + '...' if len(content) > 100 else content
            
            lines.append(
                f"{idx+1}. {artist} ({genre}) - '{title}' ({date})\n"
                f"   Brand: {brand} ({category}) | Context: {lyric}"
            )
        
        return "\n".join(lines)