"""Clean presentation script - filters out status messages for screenshots."""
import logging
import sys
from io import StringIO

# Suppress all logging
logging.disable(logging.CRITICAL)

from app.query_rag import query_system

BEST_QUERIES = [
    "Did Versace spike after Migos 'Versace' single in June 2013?",
    "Did Gucci spike after Migos Culture album in January 2017?",
    "What was the trend impact of Jay-Z's 'Tom Ford' single in July 2013?",
    "Which artists have the most diverse brand vocabulary?",
    "Which songs contain the highest number of brand references?",
]

# Lines containing these strings will be hidden
SKIP_PATTERNS = [
    "📅", "📊 Data", "✓ Brand", "✓ Enriched", "✓ Full", "✓ Taxonomy",
    "👗", "🔍", "🔄", "🎯", "📈", "📊 Found", "📊 Aggregate", 
    "📊 Comparative", "📊 SQL", "Pre:", "Post:", "Change:",
    "No significant impact", "trying multi-year", "using aggregate",
    "Fetching trends", "Auto-", "Analyzed", "Separated",
    "Artists:", "Sample brand mentions:", "artist_name", "song_title",
    "release_date", "popularity_weight", "Fashion items mentioned",
    "item_name", "Migos", "Versace", "None",  # table rows
    "----", "Type:", "Rows:",
]

def is_content_line(line: str) -> bool:
    """Return True if line should be printed."""
    stripped = line.strip()
    if not stripped:
        return False
    # Always show header lines
    if stripped.startswith("=" * 10):
        return True
    if stripped.startswith("Query:"):
        return True
    # Always show quality
    if stripped.startswith("Quality:"):
        return True
    # Always show bullet points (findings)
    if stripped.startswith("•"):
        return True
    # Skip status lines
    for pattern in SKIP_PATTERNS:
        if pattern in line:
            return False
    # Show everything else (the answer)
    return True

def run_clean(query: str):
    """Run query and print only clean output."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    try:
        query_system(query)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error: {e}")
        return
    
    # Restore stdout
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Filter and print
    lines = output.split('\n')
    for line in lines:
        if is_content_line(line):
            print(line)
    print()  # Extra spacing between queries

if __name__ == "__main__":
    for q in BEST_QUERIES:
        run_clean(q)