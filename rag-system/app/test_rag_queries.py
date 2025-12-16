import json
import time
from pathlib import Path
from typing import List, Dict, Any

from app.query_rag import query_system
from openai import OpenAI

client = OpenAI()
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluations"
EVAL_RESULTS_PATH = EVAL_DIR / "rag_evaluation_results.json"

# ------------------------------------------------------------
# 1. HARD-CODED TEST QUERIES (YOU CAN ADD OR REMOVE)
# ------------------------------------------------------------
TEST_QUERIES = [
    {
        "query": "Which year did Gucci have the most mentions in songs?",
        "expected": "Should identify the year with the highest volume of Gucci mentions."
    },
    {
        "query": "What was the most mentioned brand in 2022?",
        "expected": "Return the brand with the highest count in 2022."
    },
    {
        "query": "What brands does Travis Scott mention most?",
        "expected": "Should return brands with counts mentioned by Travis Scott."
    },
    {
        "query": "How has the popularity of boots changed over time?",
        "expected": "Should describe boots trend over years with evidence."
    }
]

# ------------------------------------------------------------
# 2. Evaluation using OpenAI LLM
# ------------------------------------------------------------
def llm_evaluate(answer: str, chunks: List[Dict], expected: str) -> Dict[str, Any]:
    """
    Uses LLM to evaluate correctness, grounding, relevance, and quality.
    """

    chunk_texts = [c.get("text", "") for c in chunks]

    prompt = f"""
You are an evaluator for RAG system answers.

User expected answer:
{expected}

System answer:
{answer}

Retrieved chunks (evidence):
{json.dumps(chunk_texts, indent=2)}

Evaluate 4 metrics from 1–10:
1. Correctness (how accurate the answer is compared to expected)
2. Faithfulness (is the answer grounded in the chunks)
3. Evidence-Relevance (are chunks relevant to the question)
4. Response Quality (clarity, completeness, reasoning)

Return strictly in JSON:
{{
    "correctness": X,
    "faithfulness": Y,
    "relevance": Z,
    "quality": Q,
    "comments": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "correctness": 0,
            "faithfulness": 0,
            "relevance": 0,
            "quality": 0,
            "comments": "⚠️ LLM response not valid JSON."
        }

# ------------------------------------------------------------
# 3. Run full evaluation loop
# ------------------------------------------------------------
def run_evaluation():
    results = []
    EVAL_DIR.mkdir(exist_ok=True)

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected"]

        print(f"\n================================================")
        print(f"Evaluating: {query}")
        print("================================================")

        # Execute your real RAG system (this calls your query_system() EXACTLY)
        rag_output = query_system(query)

        # Standardize the format
        if isinstance(rag_output, dict):
            answer = rag_output.get("answer", "")
            chunks = rag_output.get("chunks", [])
        else:
            answer = str(rag_output)
            chunks = []

        # LLM judge
        grades = llm_evaluate(answer, chunks, expected)

        results.append({
            "query": query,
            "answer": answer,
            "evaluation": grades
        })

        time.sleep(1)

    # Save results
    with EVAL_RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n🎉 Evaluation complete!")
    print(f"Results saved to {EVAL_RESULTS_PATH}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
