# app/models/trends.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class TrendsRequest(BaseModel):
    """Input parameters for Google Trends query."""
    brand: str = Field(..., description="Brand name to search in Google Trends")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    geo: str = Field("US", description="Geographic region for the search")

class TrendsDataPoint(BaseModel):
    """Single data point from Google Trends."""
    date: datetime
    value: int

class TrendsResponse(BaseModel):
    """Processed trends data with calculated metrics."""
    brand: str
    timeframe: str
    data: List[TrendsDataPoint]
    average_interest: float
    related_topics: List[str] = []
    pre_mention_avg: float = 0
    post_mention_avg: float = 0
    percent_change: float = 0