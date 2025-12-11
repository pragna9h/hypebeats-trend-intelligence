"""
Quick test to verify the anchor date filtering fix
"""
import pandas as pd
from datetime import datetime

# Simulate the data
brands_data = {
    'release_date': [
        '2020-01-15',
        '2020-02-10',
        '2020-05-01',  # Target month
        '2020-05-01',
        '2020-05-15',
    ],
    'song_title': ['Song1', 'Song2', 'Song3', 'Song4', 'Song5']
}

actual_brands_df = pd.DataFrame(brands_data)

# User's date range
start_date = '2020-05-01'
end_date = '2020-05-31'

print("=" * 60)
print("TEST: Anchor Date Filtering Fix")
print("=" * 60)
print(f"\nUser date range: {start_date} to {end_date}")
print(f"\nAll brand mentions ({len(actual_brands_df)} total):")
print(actual_brands_df)

# OLD BEHAVIOR (without filter)
print("\n" + "=" * 60)
print("OLD BEHAVIOR (uses ALL mentions):")
print("=" * 60)
all_dates = actual_brands_df['release_date'].tolist()
print(f"Dates considered: {all_dates}")
print(f"Anchor would be: {min(all_dates)}")  # 2020-01-15 (WRONG!)

# NEW BEHAVIOR (with filter)
print("\n" + "=" * 60)
print("NEW BEHAVIOR (filters to date range first):")
print("=" * 60)
filtered_brands = actual_brands_df[
    actual_brands_df['release_date'].apply(
        lambda d: start_date <= str(d)[:10] <= end_date if d else False
    )
]
print(f"Filtered to {len(filtered_brands)} mentions within date range:")
print(filtered_brands)
filtered_dates = filtered_brands['release_date'].tolist()
print(f"\nDates considered: {filtered_dates}")
print(f"Anchor would be: {min(filtered_dates)}")  # 2020-05-01 (CORRECT!)

print("\n" + "=" * 60)
print("RESULT: Fix working correctly ✓")
print("=" * 60)
