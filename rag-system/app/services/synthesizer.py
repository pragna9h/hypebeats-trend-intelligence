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
    
    SYSTEM_PROMPT = """You are an expert fashion analytics AI specializing in causal inference between hip-hop brand/item mentions and consumer trends.

DATA SOURCES:
1. Brand mentions: (artist, song, date, brand, context)
2. Lyric mentions: Full song lyrics mentioning fashion items
3. Google Trends: Brand/item search interest with pre/post metrics
   - Scale 0-100; values ≤1.0 indicate negligible search volume (not missing data)
4. Taxonomy baselines: Fashion category trends (sneakers, leather, outerwear, etc.) 
5. Monthly clusters: Grouped mentions showing cumulative momentum

ANALYSIS FRAMEWORK:

Causal Attribution:
- Primary metric: Brand/item pre-mention vs post-mention % change
- Account for 2-4 week lag between mention and peak
- Category baselines provide optional context but are NOT required for causal claims
- A brand's own pre-trend serves as its baseline
- Distinguish correlation from causation using temporal precedence

Fashion Item Analysis:
- When lyrics mention fashion items (leather, denim, boots), match to taxonomy baselines
- Analyze if item category trended after artist mentioned it
- Example: "Carti mentioned leather 5x → leather pants category +15%"

Evidence Standards:
- Cite specific examples: "Drake - 'Started From the Bottom' (2/8/2013) mentioned Nike"
- Quantify patterns: "3 mentions in March 2023 → +25% spike vs +5% sneakers baseline = +20% excess"
- For monthly clusters, track momentum: "4 mentions across Sept → pre: 45, post: 67 (+48%)"
- Note when trends preceded mentions (no causal claim possible)

Comparative Queries:
- Rank by excess impact (brand/item % - category %)
- Show absolute numbers: "Nike: +30% (sneakers baseline +5% = +25% excess)"

Fashion Trend Baselines:
- Compare item mentions (from enriched_lyrics) to taxonomy monthly averages
- Calculate 1 month pre-mention vs 2 months post-mention average
- Example: "leather mentioned 12x in Aug → taxonomy: 45 (July) → avg 52 (Aug+Sep) = +15.6%"
- Group by category (footwear, accessories, outerwear) and show top 5 by impact
- Distinguish item trends from brand trends

Multi-Year Analysis:
- When multiple years are analyzed, rank by absolute impact
- Example: "Yeezy spiked most in 2016 (+100 peak), declined in 2018 (-15%)"
- Compare year-over-year patterns when available

Data Quality:
- 'sufficient': Has pre/post trend data with temporal alignment (4+ weeks each)
- 'partial': Has trends but limited baseline (<4 weeks pre or post)
- 'insufficient': No trends data or mentions outside query timeframe
- NOTE: Category baselines optional for brand queries; brands use own pre-trend

Output:
- 2-3 sentence direct answer
- Bullet points with quantitative evidence
- Flag limitations explicitly"""

    def __init__(self):
        self.llm = LLMFactory("openai")
    
    def generate_insight(
        self,
        question: str,
        brand_mentions_df: pd.DataFrame,
        enriched_lyrics_df: pd.DataFrame,
        full_lyrics_df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        trends_data_list: list = None,
        comparative: bool = False,
        category_baseline = None,
        fashion_trends: list = None,
        aggregation_data: dict = None 
    ) -> FashionInsight:
        """Generate structured insight from brand mentions, lyrics, and trends."""
        brand_context = self._format_brand_context(brand_mentions_df)
        enriched_context = self._format_enriched_context(enriched_lyrics_df)
        lyrics_context = self._format_lyrics_context(full_lyrics_df)
        taxonomy_context = self._format_taxonomy_context(taxonomy_df)
        # SQL aggregation results (accurate counts)
        agg_context = ""
        if aggregation_data:
            agg_df = aggregation_data['aggregation_results']
            agg_context = f"""
            === SQL AGGREGATION (ACCURATE COUNTS FROM FULL DATABASE) ===
            Type: {aggregation_data['aggregation_type']}
            {aggregation_data['summary']}

            {agg_df.head(30).to_string(index=False)}

            IMPORTANT: Use these counts as PRIMARY source. They are accurate, not estimates.
            """
        
        trends_str = ""
        if trends_data_list:
            if comparative:
                trends_str = f"\n\nComparative Brand Analysis:\n"
                trends_str += f"Total brands analyzed: {len(trends_data_list)}\n\n"

                sorted_brands = sorted(
                    trends_data_list,
                    key=lambda x: x['trends'].percent_change,
                    reverse=True
                )

                for idx, brand_data in enumerate(sorted_brands, 1):
                    brand = brand_data['brand']
                    trends = brand_data['trends']
                    count = brand_data['mention_count']

                    trends_str += f"{idx}. {brand} ({count} mentions):\n"
                    trends_str += f"   Timeframe: {trends.timeframe}\n"
                    trends_str += f"   Avg interest: {trends.average_interest}\n"

                    if trends.pre_mention_avg > 0:
                        trends_str += f"   Pre-mention: {trends.pre_mention_avg}\n"
                        trends_str += f"   Post-mention: {trends.post_mention_avg}\n"
                        trends_str += f"   Change: {trends.percent_change:+.1f}%\n"

                    trends_str += "\n"

                # Add category baseline if available
                if category_baseline:
                    trends_str += f"\nCategory Baseline ({category_baseline.brand}):\n"
                    trends_str += f"   Timeframe: {category_baseline.timeframe}\n"
                    trends_str += f"   Avg interest: {category_baseline.average_interest}\n"
                    trends_str += f"   Use this to calculate excess impact: (brand % change) - (category % change)\n\n"
                    
            else:
                # Check if multi-year format (has 'year' key) vs monthly cluster format
                if trends_data_list and 'year' in trends_data_list[0]:
                    # Multi-year format
                    trends_str = f"\n\nMulti-Year Brand Analysis:\n"
                    trends_str += f"Brand: {trends_data_list[0]['trends'].brand}\n"
                    trends_str += f"Analyzed {len(trends_data_list)} years\n\n"

                    for idx, year_data in enumerate(trends_data_list, 1):
                        year = year_data['year']
                        count = year_data['mention_count']
                        trends = year_data['trends']

                        trends_str += f"{idx}. {year} ({count} mentions):\n"
                        trends_str += f"   Window: {trends.timeframe}\n"

                        if trends.pre_mention_avg > 0:
                            trends_str += f"   Pre: {trends.pre_mention_avg}, Post: {trends.post_mention_avg}\n"
                            trends_str += f"   Change: {trends.percent_change:+.1f}%\n"
                        else:
                            trends_str += f"   Avg interest: {trends.average_interest}\n"

                        if trends.data:
                            peak = max(trends.data, key=lambda x: x.value)
                            trends_str += f"   Peak: {peak.value} on {peak.date.strftime('%Y-%m-%d')}\n"

                        trends_str += "\n"
                else:
                    # Monthly cluster format
                    trends_str = f"\n\nMonthly Cluster Analysis:\n"
                    trends_str += f"Significant clusters: {len(trends_data_list)}\n\n"

                    for cluster_data in trends_data_list:
                        if 'month' in cluster_data:
                            month = cluster_data['month']
                            count = cluster_data['mention_count']
                            trends = cluster_data['trends']

                            trends_str += f"📊 {month} Cluster ({count} mentions):\n"
                            trends_str += f"   Brand: {trends.brand}\n"
                            trends_str += f"   Window: {trends.timeframe}\n"
                            trends_str += f"   Avg interest: {trends.average_interest}\n"

                            if trends.pre_mention_avg > 0:
                                trends_str += f"   Pre-cluster: {trends.pre_mention_avg}\n"
                                trends_str += f"   Post-cluster: {trends.post_mention_avg}\n"
                                trends_str += f"   Change: {trends.percent_change:+.1f}%\n"

                            if trends.data:
                                trends_str += f"   Trend timeline ({len(trends.data)} weeks):\n"
                                for point in trends.data[:3]:
                                    trends_str += f"     {point.date.strftime('%Y-%m-%d')}: {point.value}\n"
                                if len(trends.data) > 6:
                                    trends_str += f"     ...\n"
                                    for point in trends.data[-3:]:
                                        trends_str += f"     {point.date.strftime('%Y-%m-%d')}: {point.value}\n"

                            trends_str += "\n"

        fashion_trends_str = ""
        if fashion_trends:
            fashion_trends_str = "\n\nFashion Item Trend Analysis:\n"
            fashion_trends_str += f"Analyzed {len(fashion_trends)} significant items (2+ mentions)\n\n"

            for idx, trend in enumerate(fashion_trends, 1):
                fashion_trends_str += f"{idx}. {trend['item'].title()} ({trend['category']})\n"
                fashion_trends_str += f"   Mentions: {trend['mention_count']}\n"
                fashion_trends_str += f"   Baseline: {trend['pre_baseline']} → {trend['post_baseline']}\n"
                fashion_trends_str += f"   Change: {trend['percent_change']:+.1f}%\n\n"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nData:\n{brand_context}{enriched_context}{lyrics_context}{taxonomy_context}{agg_context}{trends_str}{fashion_trends_str}"}
        ]
        
        return self.llm.create_completion(
            response_model=FashionInsight,
            messages=messages,
            model="gpt-5"
        )
    
    def _format_brand_context(self, df: pd.DataFrame) -> str:
        """Format brand mentions with metadata."""
        if df.empty:
            return "No brand mentions found."

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

    def _format_enriched_context(self, df: pd.DataFrame) -> str:
        """Format enriched_lyrics showing surface form ambiguity."""
        if df.empty:
            return ""

        lines = ["\n\nFashion Item Mentions (Enriched):"]
        for idx, row in df.head(20).iterrows():
            metadata = row.get('metadata', {})
            artist = metadata.get('artist', 'Unknown')
            title = metadata.get('title', 'Unknown')
            canonical = metadata.get('canonical_label', 'Unknown')
            surface = metadata.get('surface_form', 'Unknown')

            lines.append(f"• {artist} - '{title}'")
            lines.append(f"  Said: '{surface}' → Actual item: {canonical}")
            lines.append(f"  Context: {row.get('contents', '')[:150]}...")

        return "\n".join(lines)

    def _format_lyrics_context(self, df: pd.DataFrame) -> str:
        """Format full lyrics snippets."""
        if df.empty:
            return ""

        lines = ["\n\nFull Lyrics Context:"]
        for idx, row in df.head(10).iterrows():
            metadata = row.get('metadata', {})
            artist = metadata.get('artist', 'Unknown')
            title = metadata.get('title', 'Unknown')
            lines.append(f"• {artist} - '{title}'")
            lines.append(f"  {row.get('contents', '')[:200]}...")

        return "\n".join(lines)

    def _format_taxonomy_context(self, df: pd.DataFrame) -> str:
        """Format category baseline trends."""
        if df.empty:
            return ""

        lines = ["\n\nCategory Baseline Trends:"]
        for idx, row in df.iterrows():
            import json
            stats = json.loads(row['stats']) if isinstance(row['stats'], str) else row['stats']
            lines.append(f"• {row['canonical_label']} ({row['category']})")
            lines.append(f"  Peak: {stats['peak']}, Avg: {stats['avg']:.1f}, Recent: {stats['recent_3mo_avg']:.1f}")

        return "\n".join(lines)