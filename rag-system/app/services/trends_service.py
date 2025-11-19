# app/services/trends_service.py
from pytrends.request import TrendReq
from openai import OpenAI
from pydantic import BaseModel
from datetime import datetime, timedelta
import instructor

from app.models.trends import TrendsRequest, TrendsResponse, TrendsDataPoint

class TrendsService:
    """Service for interacting with Google Trends API."""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    def get_brand_trends(
        self, 
        request: TrendsRequest, 
        mention_dates: list[str] | str | None = None
    ) -> TrendsResponse:
        """Fetch and process Google Trends data for a brand.
        
        Args:
            request: TrendsRequest with brand, dates, geo
            mention_dates: Optional date(s) when brand was mentioned (ISO format)
                          Can be single string or list. Uses first date for split.
        """
        self.pytrends.build_payload(
            [request.brand],
            timeframe=f'{request.start_date} {request.end_date}',
            geo=request.geo
        )
        df = self.pytrends.interest_over_time()
        
        if df.empty:
            raise ValueError(f"No trends data found for {request.brand}")
        
        data_points = [
            TrendsDataPoint(date=date, value=value)
            for date, value in df[request.brand].items()
        ]
        
        # Initialize pre/post metrics (defaults to 0)
        pre_avg = 0
        post_avg = 0
        pct_change = 0
        
        # Optional: Calculate pre/post if single mention date provided
        # For multiple dates, LLM will correlate temporally
        if mention_dates:
            if isinstance(mention_dates, str):
                # Single date - calculate pre/post
                mention_dt = datetime.fromisoformat(mention_dates)
                pre_data = [p for p in data_points if p.date < mention_dt]
                post_data = [p for p in data_points if p.date >= mention_dt]
                
                if pre_data and post_data:
                    pre_avg = sum(p.value for p in pre_data) / len(pre_data)
                    post_avg = sum(p.value for p in post_data) / len(post_data)
                    if pre_avg > 0:
                        pct_change = ((post_avg - pre_avg) / pre_avg) * 100
        
        return TrendsResponse(
            brand=request.brand,
            timeframe=f"{request.start_date} to {request.end_date}",
            data=data_points,
            average_interest=round(df[request.brand].mean(), 2),
            related_topics=[],
            pre_mention_avg=round(pre_avg, 2),
            post_mention_avg=round(post_avg, 2),
            percent_change=round(pct_change, 2)
        )

class TrendDecision(BaseModel):
    needs_trends: bool
    brand: str | None
    start_date: str | None
    end_date: str | None

def test_with_llm():
    client = instructor.from_openai(OpenAI())
    service = TrendsService()
    
    queries = [
        "Show Nike trends in January 2023",
        "Which brands does Drake mention?",
        "What was Gucci's popularity in 2022?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}\nQuery: {query}\n{'-'*60}")
        
        decision = client.chat.completions.create(
            model="gpt-4o",
            response_model=TrendDecision,
            messages=[{
                "role": "system",
                "content": "Determine if query needs Google Trends data and extract brand/dates."
            }, {
                "role": "user",
                "content": query
            }]
        )
        
        print(f"Needs trends: {decision.needs_trends}")
        
        if decision.needs_trends and decision.brand:
            start = decision.start_date or "2023-01-01"
            end = decision.end_date or "2023-01-31"
            
            result = service.get_brand_trends(TrendsRequest(
                brand=decision.brand,
                start_date=start,
                end_date=end
            ))
            print(f"Brand: {result.brand}")
            print(f"Avg interest: {result.average_interest}")
            print(f"Related topics: {', '.join(result.related_topics) if result.related_topics else 'None'}")

if __name__ == "__main__":
    test_with_llm()