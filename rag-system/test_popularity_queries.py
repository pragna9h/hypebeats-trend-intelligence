#!/usr/bin/env python
"""Test popularity-based trend analysis with real queries."""

from app.query_rag import query_system

# Test queries
test_queries = [
    "Did Nike spike after Drake's One Dance?",
    "Did Yeezy spike after Kanye's FACTS?",
    "What trends emerged after Future's DS2 album in 2015?",
    "Did Gucci spike after Cardi B's 7 mentions in April 2018?",
]

print("="*80)
print("TESTING POPULARITY-BASED TREND ANALYSIS")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/{len(test_queries)}: {query}")
    print("="*80)

    try:
        result = query_system(query)
        print(f"\n{result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "-"*80)

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
