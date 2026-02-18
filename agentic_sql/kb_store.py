import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_sql.memory import GlobalBusinessRulesMemory, ReasoningMemory, SchemaMemory


class KnowledgeBaseStore:
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = str(Path(db_path))
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS step_events (
                    run_id INTEGER NOT NULL,
                    step_idx INTEGER NOT NULL,
                    db_id TEXT,
                    question TEXT,
                    pred1 TEXT,
                    pred2 TEXT,
                    gold_sql TEXT,
                    lesson_text TEXT,
                    critic_trace TEXT,
                    sim1 REAL,
                    sim2 REAL,
                    pass1 INTEGER,
                    pass2 INTEGER,
                    stored_bucket TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, step_idx)
                );

                CREATE TABLE IF NOT EXISTS reasoning_lessons (
                    run_id INTEGER NOT NULL,
                    memory_id INTEGER NOT NULL,
                    title TEXT,
                    lesson_text TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, memory_id)
                );

                CREATE TABLE IF NOT EXISTS global_business_rules (
                    run_id INTEGER NOT NULL,
                    memory_id INTEGER NOT NULL,
                    title TEXT,
                    rule_text TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, memory_id)
                );

                CREATE TABLE IF NOT EXISTS schema_annotations (
                    run_id INTEGER NOT NULL,
                    db_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, db_id, entity_type, entity_name)
                );

                CREATE TABLE IF NOT EXISTS observability_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    step_idx INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def start_run(self, config: Dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO runs(config_json) VALUES (?)", (json.dumps(config),))
            return int(cur.lastrowid)

    def count_step_events(self, run_id: Optional[int] = None) -> int:
        with self._connect() as conn:
            if run_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM step_events").fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM step_events WHERE run_id=?", (run_id,)).fetchone()
        return int(row["n"]) if row is not None else 0

    def backfill_step_events_from_examples(
        self,
        examples: List[Dict[str, Any]],
        run_id: Optional[int] = None,
        source_name: str = "dataset_backfill",
    ) -> int:
        """
        Inserts raw dataset SQL examples into step_events so AST crawlers can process all queries.
        """
        rid = run_id
        if rid is None:
            rid = self.start_run(
                {
                    "mode": "dataset_backfill",
                    "source": source_name,
                    "n_examples": len(examples),
                }
            )

        rows = []
        for i, ex in enumerate(examples):
            db_id = str(ex.get("db_id") or "")
            question = str(ex.get("question") or "")
            gold_sql = str(ex.get("query") or ex.get("gold_sql") or "")
            rows.append(
                (
                    int(rid),
                    int(i),
                    db_id,
                    question,
                    "",
                    "",
                    gold_sql,
                    "",
                    "",
                    None,
                    None,
                    0,
                    0,
                    "dataset_backfill",
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO step_events(
                    run_id, step_idx, db_id, question, pred1, pred2, gold_sql,
                    lesson_text, critic_trace, sim1, sim2, pass1, pass2, stored_bucket
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_idx) DO UPDATE SET
                    db_id=excluded.db_id,
                    question=excluded.question,
                    pred1=excluded.pred1,
                    pred2=excluded.pred2,
                    gold_sql=excluded.gold_sql,
                    lesson_text=excluded.lesson_text,
                    critic_trace=excluded.critic_trace,
                    sim1=excluded.sim1,
                    sim2=excluded.sim2,
                    pass1=excluded.pass1,
                    pass2=excluded.pass2,
                    stored_bucket=excluded.stored_bucket,
                    created_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
        return int(rid)

    def latest_run_id(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                return None
            return int(row["id"])

    def write_step_event(self, run_id: int, rec: Dict[str, Any], stored_bucket: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO step_events(
                    run_id, step_idx, db_id, question, pred1, pred2, gold_sql,
                    lesson_text, critic_trace, sim1, sim2, pass1, pass2, stored_bucket
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_idx) DO UPDATE SET
                    db_id=excluded.db_id,
                    question=excluded.question,
                    pred1=excluded.pred1,
                    pred2=excluded.pred2,
                    gold_sql=excluded.gold_sql,
                    lesson_text=excluded.lesson_text,
                    critic_trace=excluded.critic_trace,
                    sim1=excluded.sim1,
                    sim2=excluded.sim2,
                    pass1=excluded.pass1,
                    pass2=excluded.pass2,
                    stored_bucket=excluded.stored_bucket,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    run_id,
                    int(rec.get("index", 0)),
                    rec.get("db_id", ""),
                    rec.get("question", ""),
                    rec.get("pred1", ""),
                    rec.get("pred2", ""),
                    rec.get("gold", ""),
                    rec.get("lesson", ""),
                    rec.get("critic_trace", ""),
                    float(rec.get("sim1", 0.0)) if rec.get("sim1") is not None else None,
                    float(rec.get("sim2", 0.0)) if rec.get("sim2") is not None else None,
                    1 if rec.get("pass1") else 0,
                    1 if rec.get("pass2") else 0,
                    stored_bucket,
                ),
            )

    def write_observability_event(
        self,
        run_id: int,
        step_idx: int,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observability_events(run_id, step_idx, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (int(run_id), int(step_idx), str(event_type), json.dumps(payload, ensure_ascii=False)),
            )

    def count_observability_events(self, run_id: Optional[int] = None) -> int:
        with self._connect() as conn:
            if run_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM observability_events").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM observability_events WHERE run_id=?",
                    (int(run_id),),
                ).fetchone()
        return int(row["n"]) if row is not None else 0

    def persist_snapshot(
        self,
        run_id: int,
        reason_memory: ReasoningMemory,
        global_memory: GlobalBusinessRulesMemory,
        schema_memory: SchemaMemory,
    ) -> None:
        reason_records = reason_memory.as_records()
        global_records = global_memory.as_records()
        schema_records = schema_memory.as_records()

        with self._connect() as conn:
            conn.execute("DELETE FROM reasoning_lessons WHERE run_id=?", (run_id,))
            conn.executemany(
                """
                INSERT INTO reasoning_lessons(run_id, memory_id, title, lesson_text, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        int(r["memory_id"]),
                        r.get("title", ""),
                        r["text"],
                        json.dumps(r.get("meta", {}), ensure_ascii=False),
                    )
                    for r in reason_records
                ],
            )

            conn.execute("DELETE FROM global_business_rules WHERE run_id=?", (run_id,))
            conn.executemany(
                """
                INSERT INTO global_business_rules(run_id, memory_id, title, rule_text, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        int(r["memory_id"]),
                        r.get("title", ""),
                        r["text"],
                        json.dumps(r.get("meta", {}), ensure_ascii=False),
                    )
                    for r in global_records
                ],
            )

            conn.execute("DELETE FROM schema_annotations WHERE run_id=?", (run_id,))
            conn.executemany(
                """
                INSERT INTO schema_annotations(run_id, db_id, entity_type, entity_name, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        r["db_id"],
                        r["entity_type"],
                        r["entity_name"],
                        r["description"],
                    )
                    for r in schema_records
                ],
            )

    def load_into_runtime_memory(
        self,
        reason_memory: ReasoningMemory,
        global_memory: GlobalBusinessRulesMemory,
        schema_memory: SchemaMemory,
        run_id: Optional[int] = None,
        clear_existing: bool = True,
    ) -> Dict[str, int]:
        """
        Rehydrates in-memory FAISS memories and schema store from SQLite snapshots.
        """
        rid = run_id if run_id is not None else self.latest_run_id()
        if rid is None:
            return {"run_id": -1, "reason_lessons": 0, "global_rules": 0, "schema_annotations": 0}

        if clear_existing:
            reason_memory.clear()
            global_memory.clear()
            schema_memory.clear()

        reason_count = 0
        global_count = 0
        schema_count = 0

        with self._connect() as conn:
            for row in conn.execute(
                """
                SELECT lesson_text, meta_json
                FROM reasoning_lessons
                WHERE run_id=?
                ORDER BY memory_id ASC
                """,
                (rid,),
            ):
                meta = json.loads(row["meta_json"] or "{}")
                reason_memory.add_lesson(row["lesson_text"], meta=meta)
                reason_count += 1

            for row in conn.execute(
                """
                SELECT rule_text, meta_json
                FROM global_business_rules
                WHERE run_id=?
                ORDER BY memory_id ASC
                """,
                (rid,),
            ):
                meta = json.loads(row["meta_json"] or "{}")
                global_memory.add_rule(row["rule_text"], meta=meta)
                global_count += 1

            for row in conn.execute(
                """
                SELECT db_id, entity_type, entity_name, description
                FROM schema_annotations
                WHERE run_id=?
                ORDER BY db_id ASC, entity_type ASC, entity_name ASC
                """,
                (rid,),
            ):
                schema_memory.upsert_annotation(
                    db_id=row["db_id"],
                    entity_type=row["entity_type"],
                    entity_name=row["entity_name"],
                    description=row["description"],
                )
                schema_count += 1

        return {
            "run_id": int(rid),
            "reason_lessons": reason_count,
            "global_rules": global_count,
            "schema_annotations": schema_count,
        }
