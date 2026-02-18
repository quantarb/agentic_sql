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

Main orchestration: `agentic_sql.pipeline`

## Key Modules

- Runtime setup (canonical init path): `agentic_sql.runtime`
- LLM orchestration: `agentic_sql.llm`
- Memory/indexing: `agentic_sql.memory`
- SQL parsing/scoring: `agentic_sql.sql_utils`
- SQLite persistence: `agentic_sql.kb_store`
- Context build/retrieval: `agentic_sql.preprocess`
- Observability sink (optional Phoenix): `agentic_sql.observability`
- Memory inspection helpers: `agentic_sql.inspection`
- Pipeline formatting helpers: `agentic_sql.pipeline_formatting`
- Pipeline memory/retrieval helpers: `agentic_sql.pipeline_memory_ops`

## Runtime Entrypoints

- Main experiment notebook: `notebook.ipynb`
- Context indexing: `examples/build_context_index.py`
- Baseline vs candidate eval: `examples/evaluate_harness.py`
- Runtime context sample feed: `examples/runtime_context.sample.jsonl`

## Defaults Worth Knowing

- Models default to `gpt-4o-mini` and embedding defaults to local `sentence_transformers` in `agentic_sql.config`.
- Use `build_runtime(...)` and `require_openai_api_key(...)` from `agentic_sql.runtime` for initialization.
- Memory governance is active; global promotion uses confidence/evidence thresholds.
- Tentative lessons can expire via TTL pruning.
- Observability is always recorded in SQLite and can be mirrored to Phoenix.

## Current Limits

1. Validation is structural (gold SQL AST), not live warehouse result validation.
2. The main loop assumes gold SQL labels exist.
3. Retrieval is not permission-aware by user identity.
4. Runtime context integration is file-based (`json/jsonl`) unless extended.
