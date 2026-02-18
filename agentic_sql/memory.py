import json
from collections import defaultdict
from typing import Any, Callable, Dict, List

import faiss
import numpy as np

from agentic_sql.sql_utils import extract_schema_from_gold_sql


def lesson_title(text: str) -> str:
    try:
        obj = json.loads(text)
        return str(obj.get("title") or obj.get("trigger") or "Untitled lesson")
    except Exception:
        return text.strip().splitlines()[0][:120] if text.strip() else "Untitled lesson"


class VectorLessonMemory:
    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray]):
        self._embed_fn = embed_fn
        self.lessons: List[str] = []
        self.meta: List[Dict[str, Any]] = []

        dim = embed_fn(["probe"]).shape[1]
        self.index = faiss.IndexFlatIP(dim)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def _build_vector(self, retrieval_key: str, text: str) -> np.ndarray:
        return self._embed_fn([f"{retrieval_key}\n{text}"])

    def clear(self) -> None:
        self.lessons = []
        self.meta = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        dim = self._embed_fn(["probe"]).shape[1]
        self.index = faiss.IndexFlatIP(dim)
        if not self.lessons:
            return

        texts = [f"{m.get('retrieval_key', '')}\n{t}" for t, m in zip(self.lessons, self.meta)]
        vecs = self._embed_fn(texts)
        self.index.add(vecs)

    def add_lesson(self, lesson_text: str, meta: Dict[str, Any]) -> int:
        key = meta.get("retrieval_key", "")
        vec = self._build_vector(key, lesson_text)
        self.index.add(vec)
        self.lessons.append(lesson_text)
        self.meta.append(meta)
        return len(self.lessons) - 1

    def as_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, (text, meta) in enumerate(zip(self.lessons, self.meta)):
            out.append(
                {
                    "memory_id": i,
                    "title": lesson_title(text),
                    "text": text,
                    "meta": dict(meta),
                }
            )
        return out

    def replace_lessons(self, ids_to_replace: List[int], merged_text: str, merged_meta: Dict[str, Any]) -> int:
        drop = set(ids_to_replace)
        self.lessons = [x for i, x in enumerate(self.lessons) if i not in drop]
        self.meta = [x for i, x in enumerate(self.meta) if i not in drop]
        self.lessons.append(merged_text)
        self.meta.append(merged_meta)
        self._rebuild_index()
        return len(self.lessons) - 1

    def remove_lessons(self, ids_to_remove: List[int]) -> int:
        drop = set(ids_to_remove)
        if not drop:
            return 0
        before = len(self.lessons)
        self.lessons = [x for i, x in enumerate(self.lessons) if i not in drop]
        self.meta = [x for i, x in enumerate(self.meta) if i not in drop]
        self._rebuild_index()
        return max(0, before - len(self.lessons))

    def retrieve_candidates(self, retrieval_key: str, k: int = 20) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        qv = self._embed_fn([retrieval_key])
        scores, ids = self.index.search(qv, min(k, self.index.ntotal))

        out = []
        for score, idx in zip(scores[0], ids[0]):
            out.append(
                {
                    "id": int(idx),
                    "text": self.lessons[idx],
                    "meta": self.meta[idx],
                    "title": lesson_title(self.lessons[idx]),
                    "score": float(score),
                }
            )
        return out

    @staticmethod
    def select_by_ids(candidates: List[Dict[str, Any]], selected_ids: List[int]) -> List[Dict[str, Any]]:
        keep = set(selected_ids)
        ordered = [c for c in candidates if int(c["id"]) in keep]
        by_id = {int(c["id"]): c for c in ordered}
        return [by_id[i] for i in selected_ids if i in by_id]

    @staticmethod
    def format(lessons: List[Dict[str, Any]], max_chars: int = 2200, label: str = "Reasoning Lesson") -> str:
        if not lessons:
            return "(none yet)"

        chunks: List[str] = []
        used = 0
        for i, lesson in enumerate(lessons, 1):
            txt = lesson["text"].strip()
            if len(txt) > 900:
                txt = txt[:900] + "..."

            chunk = f"[{label} {i} | score={lesson['score']:.3f}]\n{txt}\n"
            if used + len(chunk) > max_chars:
                break

            chunks.append(chunk)
            used += len(chunk)

        return "\n".join(chunks).strip()


class ReasoningMemory(VectorLessonMemory):
    pass


class GlobalBusinessRulesMemory(VectorLessonMemory):
    def retrieve(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.retrieve_candidates(retrieval_key=question, k=k)

    def add_rule(self, rule_text: str, meta: Dict[str, Any]) -> int:
        meta = dict(meta)
        meta["retrieval_key"] = meta.get("retrieval_key") or "global_business_rule"
        return self.add_lesson(rule_text, meta)


class SchemaMemory:
    def __init__(self):
        self._store: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(
            lambda: {"tables": {}, "columns": {}, "join_edges": {}}
        )

    def update_from_success_sql(
        self,
        db_id: str,
        question: str,
        successful_sql: str,
        annotate_fn: Callable[[str, str, str, Dict[str, List[str]]], Dict[str, Dict[str, str]]],
    ) -> None:
        feats = extract_schema_from_gold_sql(successful_sql)
        entities = {
            "tables": sorted(feats["tables"]),
            "columns": sorted(feats["columns"]),
            "join_edges": sorted(feats["join_edges"]),
        }
        annotations = annotate_fn(db_id, question, successful_sql, entities)

        bucket = self._store[db_id]
        for t in entities["tables"]:
            bucket["tables"].setdefault(t, "Seen in verified query.")
        for c in entities["columns"]:
            bucket["columns"].setdefault(c, "Used in verified query.")
        for e in entities["join_edges"]:
            bucket["join_edges"].setdefault(e, "Verified join relationship.")

        for key in ("tables", "columns", "join_edges"):
            for name, desc in annotations.get(key, {}).items():
                if name in bucket[key] and isinstance(desc, str) and desc.strip():
                    bucket[key][name] = desc.strip()

    def update_from_sql_ast(self, db_id: str, sql_text: str, source: str = "ast_bootstrap") -> bool:
        """
        Populate schema memory directly from AST-extracted identifiers without LLM annotation.
        Returns True if any new schema entries were added.
        """
        try:
            feats = extract_schema_from_gold_sql(sql_text)
        except Exception:
            return False

        before = self.count_entries(db_id)
        bucket = self._store[db_id]
        for t in feats["tables"]:
            bucket["tables"].setdefault(t, f"Discovered from SQL AST ({source}).")
        for c in feats["columns"]:
            bucket["columns"].setdefault(c, f"Discovered from SQL AST ({source}).")
        for e in feats["join_edges"]:
            bucket["join_edges"].setdefault(e, f"Discovered join edge from SQL AST ({source}).")
        after = self.count_entries(db_id)
        return after > before

    def get(self, db_id: str) -> Dict[str, Dict[str, str]]:
        return self._store[db_id]

    def count_entries(self, db_id: str) -> int:
        mem = self._store[db_id]
        return len(mem["tables"]) + len(mem["columns"]) + len(mem["join_edges"])

    def has_schema(self, db_id: str) -> bool:
        return self.count_entries(db_id) > 0

    def clear(self) -> None:
        self._store.clear()

    def upsert_annotation(self, db_id: str, entity_type: str, entity_name: str, description: str) -> None:
        key_map = {
            "table": "tables",
            "column": "columns",
            "join_edge": "join_edges",
        }
        bucket_key = key_map.get(entity_type)
        if not bucket_key:
            return
        self._store[db_id][bucket_key][entity_name] = description

    def db_ids_with_memory(self) -> List[str]:
        out = []
        for db_id, mem in self._store.items():
            if mem["tables"] or mem["columns"] or mem["join_edges"]:
                out.append(db_id)
        return out

    def format_hint(self, db_id: str, max_tables: int = 25, max_cols: int = 60, max_edges: int = 25) -> str:
        mem = self._store[db_id]
        if not (mem["tables"] or mem["columns"] or mem["join_edges"]):
            return "(none yet — you haven't verified any schema for this DB_ID)"

        tables = sorted(mem["tables"].items())[:max_tables]
        cols = sorted(mem["columns"].items())[:max_cols]
        edges = sorted(mem["join_edges"].items())[:max_edges]

        lines = [f"DB_ID={db_id}", "Known tables with descriptions:"]
        lines += [f"- {name}: {desc}" for name, desc in tables]

        lines.append("Known join edges with descriptions:")
        lines += [f"- {name}: {desc}" for name, desc in edges] if edges else ["- (none recorded yet)"]

        lines.append("Known columns with descriptions:")
        lines += [f"- {name}: {desc}" for name, desc in cols]
        return "\n".join(lines)

    def as_records(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for db_id, mem in self._store.items():
            for name, desc in mem["tables"].items():
                out.append({"db_id": db_id, "entity_type": "table", "entity_name": name, "description": desc})
            for name, desc in mem["columns"].items():
                out.append({"db_id": db_id, "entity_type": "column", "entity_name": name, "description": desc})
            for name, desc in mem["join_edges"].items():
                out.append({"db_id": db_id, "entity_type": "join_edge", "entity_name": name, "description": desc})
        return out
