import json
import os
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from sqlglot import errors as sqlglot_errors

from agentic_sql.sql_utils import extract_schema_from_gold_sql


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


class OfflineContextBuilder:
    """
    Builds a normalized multi-layer context index in SQLite.

    Layers:
    - table_usage
    - human_annotations
    - codex_enrichment
    - institutional_knowledge
    - memory
    - runtime_context (live warehouse/runtime metadata and alerts)
    """

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    db_id TEXT NOT NULL DEFAULT '',
                    entity_key TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_hash TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(layer, scope, db_id, entity_key, title)
                );

                CREATE TABLE IF NOT EXISTS context_embeddings (
                    context_id INTEGER PRIMARY KEY,
                    emb_model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _upsert_item(
        self,
        layer: str,
        scope: str,
        db_id: Optional[str],
        entity_key: Optional[str],
        title: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        payload = f"{layer}|{scope}|{db_id or ''}|{entity_key or ''}|{title}|{content}|{_json(metadata)}"
        source_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT source_hash
                FROM context_items
                WHERE layer=? AND scope=? AND db_id=? AND entity_key=? AND title=?
                """,
                (layer, scope, db_id or "", entity_key or "", title),
            ).fetchone()
            if existing is not None and str(existing["source_hash"] or "") == source_hash:
                return False
            conn.execute(
                """
                INSERT INTO context_items(layer, scope, db_id, entity_key, title, content, metadata_json, source_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(layer, scope, db_id, entity_key, title)
                DO UPDATE SET
                    content=excluded.content,
                    metadata_json=excluded.metadata_json,
                    source_hash=excluded.source_hash,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (layer, scope, db_id or "", entity_key or "", title, content, _json(metadata), source_hash),
            )
        return True

    def _safe_count(self, query: str, params: Iterable[Any] = ()) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute(query, tuple(params)).fetchone()
                return int(row[0]) if row is not None else 0
        except sqlite3.OperationalError:
            return 0

    def ingest_table_usage(self, run_id: Optional[int] = None) -> int:
        where = "WHERE run_id = ?" if run_id is not None else ""
        params: Iterable[Any] = (run_id,) if run_id is not None else ()

        count = 0
        with self._connect() as conn:
            for r in conn.execute(
                f"""
                SELECT run_id, db_id, entity_type, entity_name, description
                FROM schema_annotations
                {where}
                """,
                params,
            ):
                if self._upsert_item(
                    layer="table_usage",
                    scope="db",
                    db_id=r["db_id"],
                    entity_key=f"{r['entity_type']}:{r['entity_name']}",
                    title=f"{r['entity_type']} {r['entity_name']}",
                    content=r["description"],
                    metadata={"run_id": r["run_id"], "entity_type": r["entity_type"]},
                ):
                    count += 1
        return count

    def ingest_table_usage_from_queries(self, run_id: Optional[int] = None) -> int:
        """
        AST-crawl SQL queries from step_events and build table/column/join usage signals.
        Prefers gold_sql, then pred2, then pred1 when available.
        """
        where = "WHERE run_id = ?" if run_id is not None else ""
        params: Iterable[Any] = (run_id,) if run_id is not None else ()

        table_usage: Dict[tuple[str, str], Dict[str, Any]] = {}
        column_usage: Dict[tuple[str, str], Dict[str, Any]] = {}
        edge_usage: Dict[tuple[str, str], Dict[str, Any]] = {}
        global_edge_usage: Dict[str, Dict[str, Any]] = {}
        table_profiles: Dict[tuple[str, str], Dict[str, Any]] = {}

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, step_idx, db_id, gold_sql, pred2, pred1
                FROM step_events
                {where}
                ORDER BY run_id ASC, step_idx ASC
                """,
                params,
            ).fetchall()

        for r in rows:
            db_id = str(r["db_id"] or "")
            if not db_id:
                continue

            sql = (r["gold_sql"] or r["pred2"] or r["pred1"] or "").strip()
            if not sql:
                continue

            try:
                feats = extract_schema_from_gold_sql(sql)
            except (sqlglot_errors.ParseError, ValueError):
                continue
            except Exception:
                continue

            rid = int(r["run_id"])
            step_idx = int(r["step_idx"])

            for t in feats["tables"]:
                key = (db_id, t)
                if key not in table_usage:
                    table_usage[key] = {"count": 0, "examples": []}
                table_usage[key]["count"] += 1
                if len(table_usage[key]["examples"]) < 5:
                    table_usage[key]["examples"].append({"run_id": rid, "step_idx": step_idx})
                if key not in table_profiles:
                    table_profiles[key] = {"count": 0, "columns": set(), "join_partners": set(), "examples": []}
                table_profiles[key]["count"] += 1
                if len(table_profiles[key]["examples"]) < 5:
                    table_profiles[key]["examples"].append({"run_id": rid, "step_idx": step_idx})

            for c in feats["columns"]:
                key = (db_id, c)
                if key not in column_usage:
                    column_usage[key] = {"count": 0, "examples": []}
                column_usage[key]["count"] += 1
                if len(column_usage[key]["examples"]) < 5:
                    column_usage[key]["examples"].append({"run_id": rid, "step_idx": step_idx})
                if "." in c:
                    tname, cname = c.split(".", 1)
                    pkey = (db_id, tname)
                    if pkey not in table_profiles:
                        table_profiles[pkey] = {
                            "count": 0,
                            "columns": set(),
                            "join_partners": set(),
                            "examples": [],
                        }
                    table_profiles[pkey]["columns"].add(cname)

            for e in feats["join_edges"]:
                key = (db_id, e)
                if key not in edge_usage:
                    edge_usage[key] = {"count": 0, "examples": []}
                edge_usage[key]["count"] += 1
                if len(edge_usage[key]["examples"]) < 5:
                    edge_usage[key]["examples"].append({"run_id": rid, "step_idx": step_idx})

                if e not in global_edge_usage:
                    global_edge_usage[e] = {"count": 0, "examples": []}
                global_edge_usage[e]["count"] += 1
                if len(global_edge_usage[e]["examples"]) < 5:
                    global_edge_usage[e]["examples"].append({"run_id": rid, "step_idx": step_idx, "db_id": db_id})

                if " = " in e:
                    left, right = e.split(" = ", 1)
                    lt = left.split(".", 1)[0] if "." in left else ""
                    rt = right.split(".", 1)[0] if "." in right else ""
                    if lt and rt:
                        lkey = (db_id, lt)
                        rkey = (db_id, rt)
                        if lkey not in table_profiles:
                            table_profiles[lkey] = {
                                "count": 0,
                                "columns": set(),
                                "join_partners": set(),
                                "examples": [],
                            }
                        if rkey not in table_profiles:
                            table_profiles[rkey] = {
                                "count": 0,
                                "columns": set(),
                                "join_partners": set(),
                                "examples": [],
                            }
                        table_profiles[lkey]["join_partners"].add(rt)
                        table_profiles[rkey]["join_partners"].add(lt)

        inserted = 0
        for (db_id, table_name), payload in table_usage.items():
            if self._upsert_item(
                layer="table_usage",
                scope="db",
                db_id=db_id,
                entity_key=f"table:{table_name}",
                title=f"table {table_name}",
                content=f"Observed in {payload['count']} queries via AST crawl.",
                metadata={"source": "step_events_ast", "count": payload["count"], "examples": payload["examples"]},
            ):
                inserted += 1

        for (db_id, column_name), payload in column_usage.items():
            if self._upsert_item(
                layer="table_usage",
                scope="db",
                db_id=db_id,
                entity_key=f"column:{column_name}",
                title=f"column {column_name}",
                content=f"Observed in {payload['count']} queries via AST crawl.",
                metadata={"source": "step_events_ast", "count": payload["count"], "examples": payload["examples"]},
            ):
                inserted += 1

        for (db_id, edge_name), payload in edge_usage.items():
            if self._upsert_item(
                layer="table_usage",
                scope="db",
                db_id=db_id,
                entity_key=f"join_edge:{edge_name}",
                title=f"join_edge {edge_name}",
                content=f"Observed in {payload['count']} queries via AST crawl.",
                metadata={"source": "step_events_ast", "count": payload["count"], "examples": payload["examples"]},
            ):
                inserted += 1

        for edge_name, payload in sorted(global_edge_usage.items(), key=lambda kv: kv[1]["count"], reverse=True)[:200]:
            if self._upsert_item(
                layer="table_usage",
                scope="global",
                db_id=None,
                entity_key=f"common_join_edge:{edge_name}",
                title=f"common_join_edge {edge_name}",
                content=f"Common join pattern observed in {payload['count']} queries across DBs.",
                metadata={"source": "step_events_ast_global", "count": payload["count"], "examples": payload["examples"]},
            ):
                inserted += 1

        for (db_id, table_name), payload in table_profiles.items():
            cols = sorted(payload["columns"])[:200]
            partners = sorted(payload["join_partners"])[:200]
            content = (
                f"Auto-learned table profile from SQL AST. "
                f"Seen in {payload['count']} queries; "
                f"{len(cols)} columns observed; "
                f"{len(partners)} join partners observed."
            )
            if self._upsert_item(
                layer="table_usage",
                scope="db",
                db_id=db_id,
                entity_key=f"table_profile:{table_name}",
                title=f"table_profile {table_name}",
                content=content,
                metadata={
                    "source": "step_events_ast_profile",
                    "count": payload["count"],
                    "columns": cols,
                    "join_partners": partners,
                    "examples": payload["examples"],
                },
            ):
                inserted += 1

        return inserted

    def ingest_qa_pairs_from_queries(self, run_id: Optional[int] = None, max_pairs: int = 20000) -> int:
        where = "WHERE run_id = ?" if run_id is not None else ""
        params: Iterable[Any] = (run_id,) if run_id is not None else ()

        count = 0
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, step_idx, db_id, question, gold_sql
                FROM step_events
                {where}
                ORDER BY run_id ASC, step_idx ASC
                """,
                params,
            ).fetchall()

        for r in rows:
            q = (r["question"] or "").strip()
            sql = (r["gold_sql"] or "").strip()
            if not q or not sql:
                continue

            qa_key = hashlib.sha1(f"{r['db_id'] or ''}\n{q}\n{sql}".encode("utf-8")).hexdigest()
            if self._upsert_item(
                layer="memory",
                scope="db",
                db_id=r["db_id"] or "",
                entity_key=f"qa:{qa_key}",
                title=f"qa_pair_{qa_key[:12]}",
                content=f"Question: {q}\nSQL: {sql}",
                metadata={"source": "step_events_qa", "run_id": int(r["run_id"]), "step_idx": int(r["step_idx"])},
            ):
                count += 1
            if count >= max_pairs:
                break
        return count

    def ingest_memory(self, run_id: Optional[int] = None) -> int:
        count = 0
        with self._connect() as conn:
            if run_id is None:
                row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
                if row is None:
                    return 0
                run_id = int(row["id"])

            for r in conn.execute(
                """
                SELECT memory_id, title, lesson_text, meta_json
                FROM reasoning_lessons
                WHERE run_id=?
                """,
                (run_id,),
            ):
                if self._upsert_item(
                    layer="memory",
                    scope="db",
                    db_id=None,
                    entity_key=f"reason:{r['memory_id']}",
                    title=r["title"] or f"Reasoning lesson {r['memory_id']}",
                    content=r["lesson_text"],
                    metadata={"run_id": run_id, "meta": json.loads(r["meta_json"] or "{}")},
                ):
                    count += 1

            for r in conn.execute(
                """
                SELECT memory_id, title, rule_text, meta_json
                FROM global_business_rules
                WHERE run_id=?
                """,
                (run_id,),
            ):
                if self._upsert_item(
                    layer="memory",
                    scope="global",
                    db_id=None,
                    entity_key=f"global:{r['memory_id']}",
                    title=r["title"] or f"Global rule {r['memory_id']}",
                    content=r["rule_text"],
                    metadata={"run_id": run_id, "meta": json.loads(r["meta_json"] or "{}")},
                ):
                    count += 1
        return count

    def ingest_human_annotations(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0

        count = 0
        rows: List[Dict[str, Any]]
        if p.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            rows = json.loads(p.read_text())

        for r in rows:
            if self._upsert_item(
                layer="human_annotations",
                scope=r.get("scope", "db"),
                db_id=r.get("db_id"),
                entity_key=r.get("entity_key") or f"{r.get('entity_type','entity')}:{r.get('entity_name','unknown')}",
                title=r.get("title") or f"annotation {r.get('entity_name', 'unknown')}",
                content=r.get("content") or r.get("description") or "",
                metadata={k: v for k, v in r.items() if k not in {"content", "description", "title"}},
            ):
                count += 1
        return count

    def ingest_institutional_knowledge(self, docs_dir: str = "institutional_docs") -> int:
        root = Path(docs_dir)
        if not root.exists():
            return 0

        count = 0
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".md", ".txt"}:
                continue
            content = p.read_text(errors="ignore").strip()
            if not content:
                continue
            if self._upsert_item(
                layer="institutional_knowledge",
                scope="global",
                db_id=None,
                entity_key=str(p.relative_to(root)),
                title=p.stem,
                content=content[:8000],
                metadata={"path": str(p), "size": p.stat().st_size},
            ):
                count += 1
        return count

    def ingest_codex_enrichment(self, code_dir: str = ".") -> int:
        root = Path(code_dir)
        if not root.exists():
            return 0

        count = 0
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".sql", ".py", ".yml", ".yaml"}:
                continue
            # lightweight heuristic: only keep files that likely reference table/data logic
            txt = p.read_text(errors="ignore")
            lowered = txt.lower()
            if not any(k in lowered for k in ("select", "from ", "join ", "table", "dataset", "spark")):
                continue
            snippet = txt[:6000]
            if self._upsert_item(
                layer="codex_enrichment",
                scope="global",
                db_id=None,
                entity_key=str(p),
                title=p.name,
                content=snippet,
                metadata={"path": str(p), "size": p.stat().st_size},
            ):
                count += 1
        return count

    def ingest_runtime_context_stub(self) -> int:
        changed = self._upsert_item(
            layer="runtime_context",
            scope="global",
            db_id=None,
            entity_key="runtime_stub",
            title="runtime_query_capability",
            content=(
                "Runtime context should be fetched from live warehouse and orchestration systems "
                "when retrieved context is stale or insufficient."
            ),
            metadata={"mode": "stub"},
        )
        return 1 if changed else 0

    @staticmethod
    def _parse_iso_ts(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        t = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @classmethod
    def _runtime_is_fresh(cls, metadata: Dict[str, Any], max_age_hours: int) -> bool:
        if max_age_hours <= 0:
            return True
        now = datetime.now(timezone.utc)
        expires_at = cls._parse_iso_ts(metadata.get("expires_at"))
        if expires_at is not None:
            return expires_at >= now
        event_ts = cls._parse_iso_ts(metadata.get("event_time")) or cls._parse_iso_ts(metadata.get("updated_at"))
        if event_ts is None:
            return True
        return event_ts >= (now - timedelta(hours=max_age_hours))

    def ingest_runtime_context_live(self, path: str, max_items: int = 2000) -> int:
        """
        Ingest runtime metadata/alerts from JSON or JSONL.

        Expected per-record keys (flexible):
        - title (required-ish)
        - content (required-ish)
        - db_id, scope, entity_key
        - signal_type, severity, source, status
        - event_time (ISO), expires_at (ISO)
        - metadata (object)
        """
        p = Path(path)
        if not p.exists():
            return 0

        rows: List[Dict[str, Any]]
        if p.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            loaded = json.loads(p.read_text())
            rows = loaded if isinstance(loaded, list) else [loaded]

        count = 0
        for i, r in enumerate(rows):
            if i >= max_items:
                break
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or r.get("signal_type") or r.get("entity_key") or f"runtime_signal_{i}").strip()
            content = str(r.get("content") or r.get("description") or "").strip()
            if not content:
                continue
            scope = str(r.get("scope") or ("db" if r.get("db_id") else "global"))
            db_id = str(r.get("db_id") or "")
            entity_key = str(r.get("entity_key") or f"runtime:{r.get('signal_type','signal')}:{i}")
            metadata = {
                "mode": "live",
                "signal_type": r.get("signal_type", "runtime"),
                "severity": r.get("severity", "info"),
                "source": r.get("source", "runtime_feed"),
                "status": r.get("status", "active"),
                "event_time": r.get("event_time"),
                "expires_at": r.get("expires_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            raw_meta = r.get("metadata")
            if isinstance(raw_meta, dict):
                metadata.update(raw_meta)
            for k, v in r.items():
                if k in {
                    "title",
                    "content",
                    "description",
                    "scope",
                    "db_id",
                    "entity_key",
                    "metadata",
                    "signal_type",
                    "severity",
                    "source",
                    "status",
                    "event_time",
                    "expires_at",
                }:
                    continue
                metadata[k] = v

            if self._upsert_item(
                layer="runtime_context",
                scope=scope,
                db_id=db_id,
                entity_key=entity_key,
                title=title,
                content=content,
                metadata=metadata,
            ):
                count += 1
        return count

    def build_context(
        self,
        run_id: Optional[int] = None,
        human_annotations_path: Optional[str] = None,
        docs_dir: str = "institutional_docs",
        code_dir: str = ".",
        runtime_context_path: Optional[str] = None,
    ) -> Dict[str, int]:
        table_usage_meta = self.ingest_table_usage(run_id=run_id)
        table_usage_ast = self.ingest_table_usage_from_queries(run_id=run_id)
        qa_pairs = self.ingest_qa_pairs_from_queries(run_id=run_id)
        runtime_count = (
            self.ingest_runtime_context_live(runtime_context_path)
            if runtime_context_path and Path(runtime_context_path).exists()
            else self.ingest_runtime_context_stub()
        )
        stats = {
            "table_usage": table_usage_meta + table_usage_ast,
            "table_usage_schema_annotations": table_usage_meta,
            "table_usage_ast_crawl": table_usage_ast,
            "qa_pairs": qa_pairs,
            "memory": self.ingest_memory(run_id=run_id),
            "human_annotations": self.ingest_human_annotations(human_annotations_path)
            if human_annotations_path
            else 0,
            "codex_enrichment": self.ingest_codex_enrichment(code_dir=code_dir),
            "institutional_knowledge": self.ingest_institutional_knowledge(docs_dir=docs_dir),
            "runtime_context": runtime_count,
        }
        return stats

    def total_counts_by_layer(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT layer, COUNT(*) AS n
                FROM context_items
                GROUP BY layer
                ORDER BY layer
                """
            ).fetchall()
        for r in rows:
            out[str(r["layer"])] = int(r["n"])
        return out

    def layer_samples(self, limit_per_layer: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        counts = self.total_counts_by_layer()
        out: Dict[str, List[Dict[str, Any]]] = {}
        with self._connect() as conn:
            for layer in counts.keys():
                rows = conn.execute(
                    """
                    SELECT id, title, db_id, scope, updated_at
                    FROM context_items
                    WHERE layer=?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (layer, int(limit_per_layer)),
                ).fetchall()
                out[layer] = [
                    {
                        "id": int(r["id"]),
                        "title": str(r["title"]),
                        "db_id": str(r["db_id"] or ""),
                        "scope": str(r["scope"]),
                        "updated_at": str(r["updated_at"]),
                    }
                    for r in rows
                ]
        return out

    def source_snapshot(
        self,
        run_id: Optional[int] = None,
        human_annotations_path: Optional[str] = None,
        docs_dir: str = "institutional_docs",
        code_dir: str = ".",
    ) -> Dict[str, Any]:
        rid = run_id if run_id is not None else self._safe_count("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
        docs_root = Path(docs_dir)
        code_root = Path(code_dir)
        human_path = Path(human_annotations_path) if human_annotations_path else None

        doc_files = 0
        if docs_root.exists():
            doc_files = sum(1 for p in docs_root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"})

        candidate_code_files = 0
        if code_root.exists():
            candidate_code_files = sum(
                1 for p in code_root.rglob("*") if p.is_file() and p.suffix.lower() in {".sql", ".py", ".yml", ".yaml"}
            )

        snapshot = {
            "latest_run_id": int(rid) if rid else None,
            "step_events_rows_total": self._safe_count("SELECT COUNT(*) FROM step_events"),
            "step_events_rows_latest_run": self._safe_count(
                "SELECT COUNT(*) FROM step_events WHERE run_id=?",
                (rid,) if rid else (),
            )
            if rid
            else 0,
            "reasoning_lessons_rows": self._safe_count(
                "SELECT COUNT(*) FROM reasoning_lessons WHERE run_id=?",
                (rid,) if rid else (),
            )
            if rid
            else 0,
            "global_rules_rows": self._safe_count(
                "SELECT COUNT(*) FROM global_business_rules WHERE run_id=?",
                (rid,) if rid else (),
            )
            if rid
            else 0,
            "schema_annotations_rows": self._safe_count(
                "SELECT COUNT(*) FROM schema_annotations WHERE run_id=?",
                (rid,) if rid else (),
            )
            if rid
            else 0,
            "human_annotations_exists": bool(human_path and human_path.exists()),
            "institutional_docs_exists": docs_root.exists(),
            "institutional_doc_files": int(doc_files),
            "code_dir_exists": code_root.exists(),
            "candidate_code_files": int(candidate_code_files),
        }
        return snapshot

    def diagnostics_report(
        self,
        run_id: Optional[int] = None,
        human_annotations_path: Optional[str] = None,
        docs_dir: str = "institutional_docs",
        code_dir: str = ".",
        sample_limit: int = 3,
    ) -> Dict[str, Any]:
        return {
            "source_snapshot": self.source_snapshot(
                run_id=run_id,
                human_annotations_path=human_annotations_path,
                docs_dir=docs_dir,
                code_dir=code_dir,
            ),
            "total_counts_by_layer": self.total_counts_by_layer(),
            "layer_samples": self.layer_samples(limit_per_layer=sample_limit),
        }

    def embed_context(self, embed_fn: Callable[[List[str]], np.ndarray], emb_model: str) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.layer, c.scope, c.db_id, c.title, c.content
                FROM context_items c
                LEFT JOIN context_embeddings e ON e.context_id = c.id
                WHERE e.context_id IS NULL OR e.emb_model != ?
                ORDER BY c.id
                """,
                (emb_model,),
            ).fetchall()

        if not rows:
            return 0

        texts = [
            f"layer={r['layer']}\nscope={r['scope']}\ndb_id={r['db_id'] or ''}\ntitle={r['title']}\n{r['content']}"
            for r in rows
        ]
        # Prevent max_tokens_per_request errors by chunking embedding calls.
        batch_size = 128
        vec_chunks: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vec_chunks.append(embed_fn(batch))
        vecs = np.vstack(vec_chunks) if vec_chunks else np.zeros((0, 0), dtype=np.float32)

        with self._connect() as conn:
            for r, v in zip(rows, vecs):
                conn.execute(
                    """
                    INSERT INTO context_embeddings(context_id, emb_model, vector_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(context_id) DO UPDATE SET
                        emb_model=excluded.emb_model,
                        vector_json=excluded.vector_json,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (int(r["id"]), emb_model, _json(v.tolist())),
                )
        return len(rows)

    def retrieve(
        self,
        question: str,
        embed_fn: Callable[[List[str]], np.ndarray],
        top_k: int = 12,
        db_id: Optional[str] = None,
        layers: Optional[List[str]] = None,
        runtime_first_k: int = 0,
        runtime_boost: float = 0.0,
        runtime_max_age_hours: int = 0,
    ) -> List[Dict[str, Any]]:
        where = []
        params: List[Any] = []

        if db_id:
            where.append("(c.db_id IS NULL OR c.db_id = ?)")
            params.append(db_id)
        if layers:
            where.append("c.layer IN ({})".format(",".join(["?" for _ in layers])))
            params.extend(layers)

        clause = ("WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.layer, c.scope, c.db_id, c.title, c.content, c.metadata_json, e.vector_json
                FROM context_items c
                JOIN context_embeddings e ON e.context_id = c.id
                {clause}
                """,
                params,
            ).fetchall()

        if not rows:
            return []

        mat = np.array([json.loads(r["vector_json"]) for r in rows], dtype=np.float32)
        q = embed_fn([question]).astype(np.float32)[0]
        sims = mat @ q
        if runtime_boost:
            for idx, row in enumerate(rows):
                if str(row["layer"]) == "runtime_context":
                    sims[idx] = float(sims[idx]) + float(runtime_boost)
        order = np.argsort(-sims)[:top_k]

        def _to_item(row_idx: int) -> Dict[str, Any]:
            r = rows[int(row_idx)]
            return {
                "id": int(r["id"]),
                "layer": r["layer"],
                "scope": r["scope"],
                "db_id": r["db_id"],
                "title": r["title"],
                "content": r["content"],
                "score": float(sims[int(row_idx)]),
                "metadata": json.loads(r["metadata_json"] or "{}"),
            }

        out: List[Dict[str, Any]] = []
        picked_ids: set[int] = set()

        if runtime_first_k > 0:
            runtime_ranked = [idx for idx in order if str(rows[int(idx)]["layer"]) == "runtime_context"]
            runtime_added = 0
            for idx in runtime_ranked:
                item = _to_item(int(idx))
                if runtime_max_age_hours > 0 and not self._runtime_is_fresh(item["metadata"], runtime_max_age_hours):
                    continue
                out.append(item)
                picked_ids.add(item["id"])
                runtime_added += 1
                if runtime_added >= runtime_first_k:
                    break

        for idx in order:
            item = _to_item(int(idx))
            if item["id"] in picked_ids:
                continue
            if (
                str(item["layer"]) == "runtime_context"
                and runtime_max_age_hours > 0
                and not self._runtime_is_fresh(item["metadata"], runtime_max_age_hours)
            ):
                continue
            out.append(item)
            if len(out) >= top_k:
                break
        return out


__all__ = ["OfflineContextBuilder"]
