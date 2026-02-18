import json
import time
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from agentic_sql.config import ModelConfig
from agentic_sql.sql_utils import clean_sql


class TextToSQLLLM:
    def __init__(self, client: OpenAI, model_config: ModelConfig | None = None):
        self.client = client
        self.models = model_config or ModelConfig()
        self._local_embedder = None
        self.last_call_meta: Dict[str, Any] = {}

    _USD_PER_1M_INPUT = {
        "gpt-4o-mini": 0.15,
        "text-embedding-3-small": 0.02,
    }
    _USD_PER_1M_OUTPUT = {
        "gpt-4o-mini": 0.60,
    }

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _extract_usage(self, usage: Any) -> Dict[str, int]:
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = self._safe_int(getattr(usage, "input_tokens", None))
        output_tokens = self._safe_int(getattr(usage, "output_tokens", None))
        total_tokens = self._safe_int(getattr(usage, "total_tokens", None))
        if isinstance(usage, dict):
            input_tokens = self._safe_int(usage.get("input_tokens", input_tokens))
            output_tokens = self._safe_int(usage.get("output_tokens", output_tokens))
            total_tokens = self._safe_int(usage.get("total_tokens", total_tokens))
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": max(0, input_tokens),
            "output_tokens": max(0, output_tokens),
            "total_tokens": max(0, total_tokens),
        }

    def _estimate_cost_usd(self, model: str, usage: Dict[str, int]) -> float:
        in_rate = float(self._USD_PER_1M_INPUT.get(model, 0.0))
        out_rate = float(self._USD_PER_1M_OUTPUT.get(model, 0.0))
        in_cost = (usage["input_tokens"] / 1_000_000.0) * in_rate
        out_cost = (usage["output_tokens"] / 1_000_000.0) * out_rate
        return float(in_cost + out_cost)

    def _set_last_call_meta(
        self,
        *,
        operation: str,
        prompt_version: str,
        provider: str,
        model: str,
        latency_ms: float,
        usage: Dict[str, int],
    ) -> None:
        self.last_call_meta = {
            "operation": operation,
            "prompt_version": prompt_version,
            "provider": provider,
            "model": model,
            "latency_ms": float(latency_ms),
            "input_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "estimated_cost_usd": self._estimate_cost_usd(model=model, usage=usage),
        }

    def consume_last_call_meta(self) -> Dict[str, Any]:
        return dict(self.last_call_meta)

    def _create_response(
        self,
        *,
        model: str,
        input_payload: List[Dict[str, str]],
        operation: str,
        prompt_version: str,
    ) -> Any:
        t0 = time.perf_counter()
        resp = self.client.responses.create(model=model, input=input_payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = self._extract_usage(getattr(resp, "usage", None))
        self._set_last_call_meta(
            operation=operation,
            prompt_version=prompt_version,
            provider="openai",
            model=model,
            latency_ms=latency_ms,
            usage=usage,
        )
        return resp

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        provider = (self.models.emb_provider or "openai").lower()
        if provider == "sentence_transformers":
            if self._local_embedder is None:
                from sentence_transformers import SentenceTransformer

                self._local_embedder = SentenceTransformer(self.models.local_emb_model)
            vecs = self._local_embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            self._set_last_call_meta(
                operation="embed_texts",
                prompt_version="embed_v1",
                provider="sentence_transformers",
                model=self.models.local_emb_model,
                latency_ms=0.0,
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            return vecs

        t0 = time.perf_counter()
        resp = self.client.embeddings.create(model=self.models.emb_model, input=texts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        usage = self._extract_usage(getattr(resp, "usage", None))
        self._set_last_call_meta(
            operation="embed_texts",
            prompt_version="embed_v1",
            provider="openai",
            model=self.models.emb_model,
            latency_ms=latency_ms,
            usage=usage,
        )
        return vecs

    def generate_sql(self, question: str, db_id: str, schema_hint: str, reasoning_lessons: str) -> str:
        resp = self._create_response(
            model=self.models.gen_model,
            operation="generate_sql",
            prompt_version="sql_gen_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You are a Text-to-SQL generator.\\n"
                        "Return ONLY SQL.\\n"
                        "If schema hints are provided, treat them as reliable.\\n"
                        "If no schema hints exist, you must infer cautiously.\\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\\n\\n"
                        f"SCHEMA_HINTS (learned so far):\\n{schema_hint}\\n\\n"
                        f"REASONING_LESSONS:\\n{reasoning_lessons}\\n\\n"
                        f"QUESTION:\\n{question}\\n\\n"
                        "SQL:"
                    ),
                },
            ],
        )
        return clean_sql(resp.output_text)

    def plan_sql_approach(
        self,
        question: str,
        db_id: str,
        schema_hint: str,
        reasoning_lessons: str,
        tool_outputs: str = "",
    ) -> Dict[str, Any]:
        resp = self._create_response(
            model=self.models.planner_model,
            operation="plan_sql_approach",
            prompt_version="planner_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You are a SQL planning assistant.\n"
                        "Plan before writing SQL, using available tool outputs.\n"
                        "Return STRICT JSON with keys:\n"
                        "plan_steps: list[str]\n"
                        "likely_tables: list[str]\n"
                        "join_hypotheses: list[str]\n"
                        "filters: list[str]\n"
                        "aggregations: list[str]\n"
                        "risk_checks: list[str]\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\n\n"
                        f"QUESTION:\n{question}\n\n"
                        f"SCHEMA_HINT:\n{schema_hint}\n\n"
                        f"REASONING_LESSONS:\n{reasoning_lessons}\n\n"
                        f"TOOL_OUTPUTS:\n{tool_outputs or '(none)'}"
                    ),
                },
            ],
        )
        try:
            obj = json.loads(resp.output_text.strip())
            return {
                "plan_steps": list(obj.get("plan_steps", []))[:8],
                "likely_tables": list(obj.get("likely_tables", []))[:12],
                "join_hypotheses": list(obj.get("join_hypotheses", []))[:10],
                "filters": list(obj.get("filters", []))[:10],
                "aggregations": list(obj.get("aggregations", []))[:8],
                "risk_checks": list(obj.get("risk_checks", []))[:8],
            }
        except Exception:
            return {
                "plan_steps": [],
                "likely_tables": [],
                "join_hypotheses": [],
                "filters": [],
                "aggregations": [],
                "risk_checks": [],
            }

    @staticmethod
    def format_plan(plan: Dict[str, Any], max_chars: int = 1200) -> str:
        if not plan:
            return "(none)"
        lines: List[str] = []
        for key in ("plan_steps", "likely_tables", "join_hypotheses", "filters", "aggregations", "risk_checks"):
            vals = plan.get(key) or []
            if not vals:
                continue
            lines.append(f"{key}:")
            for v in vals:
                lines.append(f"- {str(v).strip()}")
        out = "\n".join(lines).strip() or "(none)"
        if len(out) > max_chars:
            out = out[:max_chars] + "... [truncated]"
        return out

    def synthetic_traceback(
        self, question: str, db_id: str, pred_sql: str, gold_sql: str, schema_hint: str
    ) -> str:
        resp = self._create_response(
            model=self.models.critic_model,
            operation="synthetic_traceback",
            prompt_version="critic_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You are a SQL critic acting like a simulated database engine.\n"
                        "Compare predicted SQL to gold SQL and produce a synthetic traceback.\n"
                        "Be specific about join keys, filters, grouping, and missing tables/columns.\n"
                        "Return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\n\n"
                        f"QUESTION:\n{question}\n\n"
                        f"SCHEMA_HINTS:\n{schema_hint}\n\n"
                        f"PRED_SQL:\n{pred_sql}\n\n"
                        f"GOLD_SQL:\n{gold_sql}\n\n"
                        "Write a concise synthetic traceback."
                    ),
                },
            ],
        )
        return resp.output_text.strip()

    def write_lesson(
        self,
        question: str,
        db_id: str,
        pred_sql: str,
        gold_sql: str,
        synthetic_traceback: str = "",
    ) -> str:
        resp = self._create_response(
            model=self.models.lesson_model,
            operation="write_lesson",
            prompt_version="lesson_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "Write a reusable lesson that will help answer future Text-to-SQL questions.\\n"
                        "Return STRICT JSON ONLY.\\n"
                        "Make it generally applicable; if schema was the issue, include a schema section inferred from GOLD SQL.\\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\\n\\n"
                        f"QUESTION:\\n{question}\\n\\n"
                        f"PRED_SQL (wrong):\\n{pred_sql}\\n\\n"
                        f"GOLD_SQL (correct):\\n{gold_sql}\\n\\n"
                        f"SYNTHETIC_TRACEBACK:\\n{synthetic_traceback or '(none)'}\\n\\n"
                        "Return JSON with keys:\\n"
                        "- title\\n- trigger\\n- diagnosis\\n- fix_rule\\n"
                        "- schema_guess (optional: tables, columns, join_edges)\\n"
                        "- scope (optional: 'db_specific' or 'global')\\n"
                        "- example (optional)\\n"
                    ),
                },
            ],
        )
        return resp.output_text.strip()

    def rerank_lesson_ids(
        self,
        question: str,
        db_id: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[int]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return [int(c["id"]) for c in candidates]

        compact = []
        for c in candidates:
            compact.append(
                {
                    "id": c["id"],
                    "title": c.get("title", ""),
                    "score": c.get("score", 0.0),
                }
            )

        resp = self._create_response(
            model=self.models.reranker_model,
            operation="rerank_lesson_ids",
            prompt_version="reranker_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You rerank candidate lessons for Text-to-SQL.\n"
                        "Return STRICT JSON: {\"selected_ids\": [int, ...]}.\n"
                        "Pick the most logically relevant lessons for the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\n\n"
                        f"QUESTION:\n{question}\n\n"
                        f"CANDIDATES:\n{json.dumps(compact, ensure_ascii=False)}\n\n"
                        f"Return exactly {top_k} IDs if available."
                    ),
                },
            ],
        )
        try:
            obj = json.loads(resp.output_text.strip())
            ids = [int(x) for x in obj.get("selected_ids", [])]
            id_set = {int(c["id"]) for c in candidates}
            chosen = [x for x in ids if x in id_set]
            if not chosen:
                return [int(c["id"]) for c in candidates[:top_k]]
            return chosen[:top_k]
        except Exception:
            return [int(c["id"]) for c in candidates[:top_k]]

    def consolidate_candidate_lessons(
        self,
        new_lesson: str,
        similar_lessons: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resp = self._create_response(
            model=self.models.consolidator_model,
            operation="consolidate_candidate_lessons",
            prompt_version="consolidator_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory manager for SQL lessons.\n"
                        "Decide whether a new lesson is duplicate/overlapping with existing lessons.\n"
                        "Return STRICT JSON with keys:\n"
                        "action: 'merge' | 'keep'\n"
                        "merged_text: string\n"
                        "merge_ids: list[int] (existing IDs to replace if action='merge')\n"
                        "scope: 'db_specific' | 'global'"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"NEW_LESSON:\n{new_lesson}\n\n"
                        f"SIMILAR_EXISTING_LESSONS:\n{json.dumps(similar_lessons, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        try:
            out = json.loads(resp.output_text.strip())
            return {
                "action": out.get("action", "keep"),
                "merged_text": out.get("merged_text", new_lesson),
                "merge_ids": [int(x) for x in out.get("merge_ids", [])],
                "scope": out.get("scope", "db_specific"),
            }
        except Exception:
            return {
                "action": "keep",
                "merged_text": new_lesson,
                "merge_ids": [],
                "scope": "db_specific",
            }

    def annotate_schema_from_success(
        self,
        db_id: str,
        question: str,
        sql: str,
        entities: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, str]]:
        resp = self._create_response(
            model=self.models.schema_model,
            operation="annotate_schema_from_success",
            prompt_version="schema_annotate_v1",
            input_payload=[
                {
                    "role": "system",
                    "content": (
                        "You generate semantic schema annotations from a successful SQL query.\n"
                        "Return STRICT JSON with keys: tables, columns, join_edges.\n"
                        "Each key maps names to concise descriptions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DB_ID: {db_id}\n\n"
                        f"QUESTION:\n{question}\n\n"
                        f"SUCCESSFUL_SQL:\n{sql}\n\n"
                        f"ENTITIES:\n{json.dumps(entities, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        try:
            raw = json.loads(resp.output_text.strip())
            return {
                "tables": dict(raw.get("tables", {})),
                "columns": dict(raw.get("columns", {})),
                "join_edges": dict(raw.get("join_edges", {})),
            }
        except Exception:
            return {"tables": {}, "columns": {}, "join_edges": {}}

    @staticmethod
    def normalize_lesson(lesson_raw: str) -> str:
        try:
            lesson_obj: Dict[str, Any] = json.loads(lesson_raw)
            return json.dumps(lesson_obj, ensure_ascii=False)
        except Exception:
            return lesson_raw
