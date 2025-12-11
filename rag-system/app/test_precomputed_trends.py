#!/usr/bin/env python
"""Test script to verify pre-computed brand trends integration."""

import logging
from app.services.trends_service import TrendsService, BrandNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_precomputed_trends():
    """Test fetching pre-computed trends for a known brand."""
    service = TrendsService()

    # Test with a brand we know is in the data (Balenciaga, Dior, etc.)
    test_brand = "balenciaga"
    start_date = "2015-01-01"
    end_date = "2015-06-30"

    print(f"\n{'='*60}")
    print(f"Testing pre-computed trends for '{test_brand}'")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    try:
        result = service.get_brand_trends_from_precomputed(
            brand=test_brand,
            start_date=start_date,
            end_date=end_date,
            mention_dates="2015-03-15"  # Test split point
        )

        print(f"✓ Successfully fetched pre-computed data!")
        print(f"\nBrand: {result.brand}")
        print(f"Timeframe: {result.timeframe}")
        print(f"Data points: {len(result.data)}")
        print(f"Average interest: {result.average_interest}")
        print(f"Pre-mention avg: {result.pre_mention_avg}")
        print(f"Post-mention avg: {result.post_mention_avg}")
        print(f"Percent change: {result.percent_change:+.1f}%")

        print(f"\nFirst 3 data points:")
        for dp in result.data[:3]:
            print(f"  {dp.date.strftime('%Y-%m-%d')}: {dp.value}")

    except BrandNotFoundError as e:
        print(f"❌ Brand not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with non-existent brand
    print(f"\n{'='*60}")
    print(f"Testing fallback with non-existent brand")
    print(f"{'='*60}\n")

    try:
        result = service.get_brand_trends_from_precomputed(
            brand="nonexistentbrand123",
            start_date="2020-01-01",
            end_date="2020-06-30"
        )
        print(f"❌ Should have raised BrandNotFoundError")
        return False
    except BrandNotFoundError as e:
        print(f"✓ Correctly raised BrandNotFoundError: {e}")

    print(f"\n{'='*60}")
    print(f"✅ All tests passed!")
    print(f"{'='*60}\n")
    return True

if __name__ == "__main__":
    test_precomputed_trends()
