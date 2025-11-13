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
- Mark data_quality as 'insufficient' if can't answer properly"""

    def __init__(self):
        self.llm = LLMFactory("openai")
    
    def generate_insight(self, question: str, context_df: pd.DataFrame) -> FashionInsight:
        """Generate structured insight from retrieved mentions."""
        context_str = self._format_context(context_df)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nData:\n{context_str}"}
        ]
        
        return self.llm.create_completion(
            response_model=FashionInsight,
            messages=messages,
            model="gpt-4o"
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
            
            # Extract lyric context
            content = row.get('contents', '')
            lyric = content[:100] + '...' if len(content) > 100 else content
            
            lines.append(
                f"{idx+1}. {artist} ({genre}) - '{title}' ({date})\n"
                f"   Brand: {brand} ({category}) | Context: {lyric}"
            )
        
        return "\n".join(lines)