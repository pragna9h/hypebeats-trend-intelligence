# test_research_queries.py
# Run this to test all RQ-relevant queries and save outputs

"""
Usage: python -m app.test_research_queries
This will write a full log to rag-system/evaluations/research_results.txt
"""

import sys
from pathlib import Path

from app.query_rag import query_system

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluations"
RESEARCH_LOG_PATH = EVAL_DIR / "research_results.txt"


class DualWriter:
    """Write stdout to console and file simultaneously."""

    def __init__(self, *targets):
        self.targets = targets

    def write(self, data):
        for target in self.targets:
            target.write(data)

    def flush(self):
        for target in self.targets:
            target.flush()

# =============================================================================
# RQ1: Brand mentions → Google search interest (lag analysis)
# =============================================================================
RQ1_QUERIES = [
    # Known cultural moments (should show clear spikes)
    "Did Versace spike after Migos 'Versace' single in June 2013?",
    "What was the trend impact of Jay-Z's 'Tom Ford' single in July 2013?",
    "Did Balenciaga spike after Cardi B mentions in 2018?",
    
    # Null hypothesis tests (should show NO spike or decline)
    "Did Gucci spike after Migos Culture album in January 2017?",
    
    # Multi-year trajectory (shows lag patterns over time)
    "What is the influence trajectory of Versace brand from 2013 to 2023?",
    "Generate a marketing insight report for 'Louis Vuitton' based on all mentions.",
    
    # Comparative impact
    "Compare Gucci vs Versace trend impact across all mentions.",
]

# =============================================================================
# RQ2: System capability - natural language questions
# =============================================================================
RQ2_QUERIES = [
    # Aggregation queries (SQL path)
    "Which artists have the most diverse brand vocabulary?",
    "Which songs contain the highest number of brand references?",
    "For artist Future, what are the top brands referenced across his discography?",
    
    # Ranking/impact queries
    "Which brands show the strongest post-mention search increases?",
    "Rank the top 5 artists by their influence on luxury brand search trends.",
    
    # Comparative
    "Which artist drove bigger brand search impact: Future or Young Thug?",
    "Compare Nike vs Adidas mentions and trend impact.",
]

# =============================================================================
# RQ3: Strengths and failure modes
# =============================================================================
RQ3_QUERIES = [
    # Self-referential (brand = artist name)
    "Did Yeezy spike after Kanye West mentions?",
    
    # Fashion items vs brands
    "What fashion items does Playboi Carti mention in his lyrics?",
    
    # Watch brands (luxury items)
    "Did Rolex spike after hip-hop mentions in 2018?",
    "What is the trend impact of Patek Philippe mentions?",
    
    # Ambiguous brand (Guess = common word)
    "How many times is Guess brand mentioned vs the word 'guess'?",
    
    # Limited data scenarios
    "Did Zara spike after any hip-hop mentions?",
]

def run_test_suite():
    print("=" * 100)
    print("HYPEBEATS RAG SYSTEM - RESEARCH QUESTION TEST SUITE")
    print("=" * 100)
    
    all_queries = [
        ("RQ1: Lag Analysis & Causal Impact", RQ1_QUERIES),
        ("RQ2: System Capability Demonstration", RQ2_QUERIES),
        ("RQ3: Strengths & Failure Modes", RQ3_QUERIES),
    ]
    
    results_summary = []
    
    for section_name, queries in all_queries:
        print(f"\n{'#' * 100}")
        print(f"# {section_name}")
        print(f"{'#' * 100}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{section_name[:3]} Query {i}/{len(queries)}]")
            try:
                query_system(query)
            except Exception as e:
                print(f"ERROR: {e}")
                results_summary.append((query, "ERROR", str(e)))
            
            print("\n" + "-" * 100)
    
    # Print summary at end
    print("\n" + "=" * 100)
    print("TEST SUITE COMPLETE")
    print("=" * 100)
    print(f"Total queries run: {len(RQ1_QUERIES) + len(RQ2_QUERIES) + len(RQ3_QUERIES)}")
    print("\nReview the output above to categorize:")
    print("  - SUFFICIENT quality responses")
    print("  - PARTIAL quality responses") 
    print("  - INSUFFICIENT quality responses")
    print("  - Clear spike detections vs null results")


if __name__ == "__main__":
    EVAL_DIR.mkdir(exist_ok=True)
    with RESEARCH_LOG_PATH.open("w") as log_file:
        dual_writer = DualWriter(sys.stdout, log_file)
        original_stdout = sys.stdout
        try:
            sys.stdout = dual_writer
            run_test_suite()
            print(f"\nSaved detailed log to {RESEARCH_LOG_PATH}")
        finally:
            sys.stdout = original_stdout
