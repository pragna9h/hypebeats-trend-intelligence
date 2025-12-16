# Evaluation Artifacts

This folder collects reproducible evaluation outputs so metrics stay in one place.

- `rag_evaluation_results.json` – LLM-graded answers for the canned queries in `app/test_rag_queries.py`.
- `research_results.txt` – Full transcript from the research question suite in `app/test_research_queries.py`.

## How to regenerate

From `rag-system/`:

```bash
# LLM-graded spot checks
python -m app.test_rag_queries

# Full research question run (logs to this folder automatically)
python -m app.test_research_queries
```

Both scripts create this directory if it does not exist and write their outputs here.
