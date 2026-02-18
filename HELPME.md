# Agentic SQL: High-Level Guide

Last updated: 2026-02-18

This project is a self-improving Text-to-SQL agent. It generates SQL from natural language, evaluates against gold SQL with AST-aware scoring, learns lessons, retries, and persists memory/telemetry to SQLite.

## End-to-End Flow

1. Input: question + `db_id` + gold SQL.
2. Retrieve context:
- DB-specific reasoning lessons.
- Global rules.
- Optional offline/runtime context (`OfflineContextBuilder`).
3. Generate first SQL attempt.
4. Score with AST-aware metrics (`exact`, `structural_similarity`).
5. On failure:
- Generate critic feedback.
- Write structured lesson.
- Retry generation with that lesson.
6. On success:
- Update schema memory.
- Store lesson as DB-specific or global (gated by confidence/evidence).
7. Persist everything to `knowledge_base.db`:
- Runs, step events, memory snapshots, schema annotations, observability events.

Main orchestration: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/pipeline.py`

## Key Modules

- Runtime setup (canonical init path): `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/runtime.py`
- LLM orchestration: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/llm.py`
- Memory/indexing: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/memory.py`
- SQL parsing/scoring: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/sql_utils.py`
- SQLite persistence: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/kb_store.py`
- Context build/retrieval: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/preprocess.py`
- Observability sink (optional Phoenix): `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/observability.py`
- Memory inspection helpers: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/inspection.py`
- Pipeline formatting helpers: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/pipeline_formatting.py`
- Pipeline memory/retrieval helpers: `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/pipeline_memory_ops.py`

## Runtime Entrypoints

- Main experiment notebook: `/Users/johnnylee/PycharmProjects/agentic_sql/notebook.ipynb`
- Context indexing: `/Users/johnnylee/PycharmProjects/agentic_sql/examples/build_context_index.py`
- Baseline vs candidate eval: `/Users/johnnylee/PycharmProjects/agentic_sql/examples/evaluate_harness.py`
- Runtime context sample feed: `/Users/johnnylee/PycharmProjects/agentic_sql/examples/runtime_context.sample.jsonl`

## Defaults Worth Knowing

- Models default to `gpt-4o-mini` and embedding defaults to local `sentence_transformers` in `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/config.py`.
- Use `build_runtime(...)` and `require_openai_api_key(...)` from `/Users/johnnylee/PycharmProjects/agentic_sql/agentic_sql/runtime.py` for initialization.
- Memory governance is active; global promotion uses confidence/evidence thresholds.
- Tentative lessons can expire via TTL pruning.
- Observability is always recorded in SQLite and can be mirrored to Phoenix.

## Current Limits

1. Validation is structural (gold SQL AST), not live warehouse result validation.
2. The main loop assumes gold SQL labels exist.
3. Retrieval is not permission-aware by user identity.
4. Runtime context integration is file-based (`json/jsonl`) unless extended.
