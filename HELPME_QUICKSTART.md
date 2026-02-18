# Agentic SQL Quickstart (5 Minutes)

Last updated: 2026-02-18

## 1. Install dependencies

```bash
pip install openai datasets sqlglot faiss-cpu sentence-transformers numpy
```

Optional (Phoenix export):

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

## 2. Set env vars

```bash
export OPENAI_API_KEY="YOUR_KEY"
export RUNTIME_CONTEXT_PATH="/Users/johnnylee/PycharmProjects/agentic_sql/examples/runtime_context.sample.jsonl"
```

Optional Phoenix:

```bash
export PHOENIX_ENABLED=1
export PHOENIX_PROJECT_NAME=agentic-sql
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
```

## 3. Build context index

```bash
python /Users/johnnylee/PycharmProjects/agentic_sql/examples/build_context_index.py
```

This ingests context layers and embeds them into `knowledge_base.db`.

## 4. Run training/eval loop (notebook)

Open and run all cells in:
- `/Users/johnnylee/PycharmProjects/agentic_sql/notebook.ipynb`

This runs generate -> score -> learn -> retry and writes run + observability data into SQLite.
Notebook uses the canonical runtime setup helpers (`build_runtime`, `require_openai_api_key`).

## 5. Run the eval harness

```bash
python /Users/johnnylee/PycharmProjects/agentic_sql/examples/evaluate_harness.py \
  --split train \
  --n-steps 50 \
  --output-json /Users/johnnylee/PycharmProjects/agentic_sql/eval_report.json \
  --output-md /Users/johnnylee/PycharmProjects/agentic_sql/eval_report.md
```

This compares `baseline_no_memory` vs `candidate_default` and applies gating thresholds.

## 6. Quick verification

- SQLite DB: `/Users/johnnylee/PycharmProjects/agentic_sql/knowledge_base.db`
- Report JSON: `/Users/johnnylee/PycharmProjects/agentic_sql/eval_report.json`
- Report Markdown: `/Users/johnnylee/PycharmProjects/agentic_sql/eval_report.md`

Useful SQL checks:

```sql
SELECT event_type, COUNT(*) AS n
FROM observability_events
GROUP BY event_type
ORDER BY n DESC;
```

```sql
SELECT layer, COUNT(*) AS n
FROM context_items
GROUP BY layer
ORDER BY layer;
```

## Notes

- If Phoenix deps are missing, SQLite observability still works.
- `OPENAI_API_KEY` is required by the example scripts.
- Default embeddings are local sentence-transformers unless model config is changed.
- Runtime initialization is standardized in `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/runtime.py`.
