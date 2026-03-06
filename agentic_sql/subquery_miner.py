import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sqlglot
from sqlglot import exp

from agentic_sql.sql_utils import clean_sql, strip_literals_in_ast


def _hash_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _sql_id(kind: str, canonical_sql: str) -> str:
    return _hash_id(f"{kind}|{canonical_sql}")


def _fold_binary_chain(items: Sequence[exp.Expression], node_type: type) -> exp.Expression:
    if not items:
        raise ValueError("Expected at least one item")
    cur = items[0]
    for nxt in items[1:]:
        cur = node_type(this=cur, expression=nxt)
    return cur


def _flatten_binary(node: exp.Expression, node_type: type) -> List[exp.Expression]:
    if not isinstance(node, node_type):
        return [node]
    left = _flatten_binary(node.this, node_type) if isinstance(node.this, exp.Expression) else []
    right = _flatten_binary(node.expression, node_type) if isinstance(node.expression, exp.Expression) else []
    return left + right


def _split_conjuncts(node: exp.Expression) -> List[exp.Expression]:
    return _flatten_binary(node, exp.And)


class SQLCanonicalizer:
    """
    Canonicalizes SQL queries and expressions so equivalent structures map together.
    """

    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect

    def parse(self, sql: str) -> exp.Expression:
        return sqlglot.parse_one(clean_sql(sql), read=self.dialect)

    def canonicalize_sql(self, sql: str) -> str:
        ast = self.parse(sql)
        return self.canonicalize_node(ast)

    def canonicalize_node(self, node: exp.Expression) -> str:
        n = strip_literals_in_ast(node.copy())
        n = self._normalize_aliases(n)
        n = self._normalize_commutative_expressions(n)
        return n.sql(dialect=self.dialect, normalize=True)

    def _normalize_aliases(self, node: exp.Expression) -> exp.Expression:
        alias_map: Dict[str, str] = {}
        alias_idx = 0

        for table in node.find_all(exp.Table):
            table_alias = table.args.get("alias")
            if isinstance(table_alias, exp.TableAlias) and isinstance(table_alias.this, exp.Identifier):
                old_alias = table_alias.this.name
                if old_alias not in alias_map:
                    alias_idx += 1
                    alias_map[old_alias] = f"t{alias_idx}"
                table_alias.set("this", exp.to_identifier(alias_map[old_alias]))

        for col in node.find_all(exp.Column):
            tbl = col.table
            if tbl and tbl in alias_map:
                col.set("table", exp.to_identifier(alias_map[tbl]))

        return node

    def _normalize_commutative_expressions(self, node: exp.Expression) -> exp.Expression:
        for sub in list(node.walk()):
            if isinstance(sub, exp.And):
                terms = [_ for _ in _split_conjuncts(sub) if isinstance(_, exp.Expression)]
                normalized = [self._normalize_commutative_expressions(t.copy()) for t in terms]
                normalized.sort(key=lambda t: t.sql(dialect=self.dialect, normalize=True))
                rebuilt = _fold_binary_chain(normalized, exp.And)
                sub.replace(rebuilt)
                continue

            if isinstance(sub, exp.Or):
                terms = [_ for _ in _flatten_binary(sub, exp.Or) if isinstance(_, exp.Expression)]
                normalized = [self._normalize_commutative_expressions(t.copy()) for t in terms]
                normalized.sort(key=lambda t: t.sql(dialect=self.dialect, normalize=True))
                rebuilt = _fold_binary_chain(normalized, exp.Or)
                sub.replace(rebuilt)
                continue

            if isinstance(sub, exp.EQ):
                left = sub.left
                right = sub.right
                if isinstance(left, exp.Expression) and isinstance(right, exp.Expression):
                    ln = self._normalize_commutative_expressions(left.copy())
                    rn = self._normalize_commutative_expressions(right.copy())
                    ordered = sorted(
                        [ln, rn], key=lambda t: t.sql(dialect=self.dialect, normalize=True)
                    )
                    sub.set("this", ordered[0])
                    sub.set("expression", ordered[1])
                continue

            if isinstance(sub, exp.In):
                exprs = sub.expressions
                if exprs:
                    normalized = [self._normalize_commutative_expressions(x.copy()) for x in exprs]
                    normalized.sort(key=lambda t: t.sql(dialect=self.dialect, normalize=True))
                    sub.set("expressions", normalized)
                continue
        return node


class SQLSubplanExtractor:
    """
    Extracts reusable SQL fragments from parsed SQL trees.
    """

    def __init__(self, canonicalizer: Optional[SQLCanonicalizer] = None):
        self.canonicalizer = canonicalizer or SQLCanonicalizer()

    def extract(self, ast: exp.Expression) -> List[Tuple[str, str]]:
        fragments: List[Tuple[str, str]] = []
        for sel in ast.find_all(exp.Select):
            fragments.append(("select_block", self.canonicalizer.canonicalize_node(sel)))

            from_clause = sel.args.get("from")
            if isinstance(from_clause, exp.From):
                fragments.append(("from_clause", self.canonicalizer.canonicalize_node(from_clause)))

            joins = sel.args.get("joins") or []
            for join in joins:
                if isinstance(join, exp.Join):
                    fragments.append(("join_clause", self.canonicalizer.canonicalize_node(join)))
                    join_on = join.args.get("on")
                    if isinstance(join_on, exp.Expression):
                        for pred in _split_conjuncts(join_on):
                            fragments.append(
                                ("join_predicate", self.canonicalizer.canonicalize_node(pred))
                            )

            where_clause = sel.args.get("where")
            if isinstance(where_clause, exp.Where) and isinstance(where_clause.this, exp.Expression):
                for pred in _split_conjuncts(where_clause.this):
                    fragments.append(("where_filter", self.canonicalizer.canonicalize_node(pred)))

            having_clause = sel.args.get("having")
            if isinstance(having_clause, exp.Having) and isinstance(having_clause.this, exp.Expression):
                for pred in _split_conjuncts(having_clause.this):
                    fragments.append(("having_filter", self.canonicalizer.canonicalize_node(pred)))

            group_clause = sel.args.get("group")
            if isinstance(group_clause, exp.Group):
                for group_expr in group_clause.expressions:
                    if isinstance(group_expr, exp.Expression):
                        fragments.append(
                            ("group_key", self.canonicalizer.canonicalize_node(group_expr))
                        )

            order_clause = sel.args.get("order")
            if isinstance(order_clause, exp.Order):
                for ordered in order_clause.expressions:
                    if isinstance(ordered, exp.Expression):
                        fragments.append(
                            ("order_key", self.canonicalizer.canonicalize_node(ordered))
                        )

        for subquery in ast.find_all(exp.Subquery):
            if isinstance(subquery.this, exp.Expression):
                fragments.append(("subquery", self.canonicalizer.canonicalize_node(subquery.this)))

        return fragments

    def extract_from_sql(self, sql: str) -> Tuple[str, List[Tuple[str, str]]]:
        ast = self.canonicalizer.parse(sql)
        canonical_query = self.canonicalizer.canonicalize_node(ast)
        fragments = self.extract(ast)
        return canonical_query, fragments


@dataclass
class IngestStats:
    processed: int
    parsed: int
    failed: int


class SQLPatternMiner:
    """
    Disk-backed SQL pattern discovery:
    - canonical query/fragment extraction
    - bipartite graph materialization (query <-> fragment)
    - PageRank and support metrics over the graph
    """

    def __init__(self, db_path: str = "sql_pattern_mining.db", dialect: str = "sqlite"):
        self.db_path = db_path
        self.extractor = SQLSubplanExtractor(SQLCanonicalizer(dialect=dialect))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    canonical_sql TEXT NOT NULL,
                    raw_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS fragments (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    canonical_sql TEXT NOT NULL,
                    raw_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS query_fragment_edges (
                    query_id TEXT NOT NULL,
                    fragment_id TEXT NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (query_id, fragment_id),
                    FOREIGN KEY (query_id) REFERENCES queries(id),
                    FOREIGN KEY (fragment_id) REFERENCES fragments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_qf_fragment ON query_fragment_edges(fragment_id);
                CREATE INDEX IF NOT EXISTS idx_qf_query ON query_fragment_edges(query_id);
                CREATE INDEX IF NOT EXISTS idx_frag_kind ON fragments(kind);

                CREATE TABLE IF NOT EXISTS fragment_metrics (
                    fragment_id TEXT PRIMARY KEY,
                    pagerank REAL NOT NULL,
                    query_support INTEGER NOT NULL,
                    raw_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_metrics (
                    query_id TEXT PRIMARY KEY,
                    pagerank REAL NOT NULL,
                    fragment_degree INTEGER NOT NULL,
                    raw_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                );
                """
            )

    def ingest_queries(self, sql_queries: Iterable[str], batch_size: int = 1000) -> IngestStats:
        processed = parsed = failed = 0
        query_rows: List[Tuple[str, str]] = []
        fragment_rows: List[Tuple[str, str, str, int]] = []
        edge_rows: List[Tuple[str, str, int]] = []

        def flush() -> None:
            if not query_rows and not fragment_rows and not edge_rows:
                return
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO queries(id, canonical_sql, raw_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(id) DO UPDATE SET raw_count = raw_count + 1
                    """,
                    query_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO fragments(id, kind, canonical_sql, raw_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET raw_count = raw_count + excluded.raw_count
                    """,
                    fragment_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO query_fragment_edges(query_id, fragment_id, weight)
                    VALUES (?, ?, ?)
                    ON CONFLICT(query_id, fragment_id) DO UPDATE SET weight = weight + excluded.weight
                    """,
                    edge_rows,
                )
            query_rows.clear()
            fragment_rows.clear()
            edge_rows.clear()

        for sql in sql_queries:
            processed += 1
            try:
                canonical_query, fragments = self.extractor.extract_from_sql(sql)
                parsed += 1
            except Exception:
                failed += 1
                continue

            qid = _hash_id(canonical_query)
            query_rows.append((qid, canonical_query))

            frag_counter = Counter()
            frag_meta: Dict[str, Tuple[str, str]] = {}
            for kind, frag_sql in fragments:
                fid = _sql_id(kind, frag_sql)
                frag_counter[fid] += 1
                frag_meta[fid] = (kind, frag_sql)

            for fid, cnt in frag_counter.items():
                kind, frag_sql = frag_meta[fid]
                fragment_rows.append((fid, kind, frag_sql, cnt))
                edge_rows.append((qid, fid, cnt))

            if processed % batch_size == 0:
                flush()

        flush()
        return IngestStats(processed=processed, parsed=parsed, failed=failed)

    def run_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 30,
        tol: float = 1e-8,
    ) -> Dict[str, float]:
        with self._connect() as conn:
            n_queries = conn.execute("SELECT COUNT(*) AS c FROM queries").fetchone()["c"]
            n_fragments = conn.execute("SELECT COUNT(*) AS c FROM fragments").fetchone()["c"]
            total_nodes = n_queries + n_fragments
            if total_nodes == 0:
                return {"iterations": 0, "delta": 0.0}

            base = (1.0 - damping) / float(total_nodes)
            init_rank = 1.0 / float(total_nodes)

            conn.executescript(
                """
                DROP TABLE IF EXISTS q_degree;
                DROP TABLE IF EXISTS f_degree;
                DROP TABLE IF EXISTS q_rank;
                DROP TABLE IF EXISTS f_rank;

                CREATE TEMP TABLE q_degree AS
                SELECT query_id, SUM(weight) AS degree
                FROM query_fragment_edges
                GROUP BY query_id;

                CREATE TEMP TABLE f_degree AS
                SELECT fragment_id, SUM(weight) AS degree
                FROM query_fragment_edges
                GROUP BY fragment_id;
                """
            )

            conn.execute("CREATE TEMP TABLE q_rank(query_id TEXT PRIMARY KEY, rank REAL NOT NULL)")
            conn.execute("CREATE TEMP TABLE f_rank(fragment_id TEXT PRIMARY KEY, rank REAL NOT NULL)")
            conn.execute("INSERT INTO q_rank(query_id, rank) SELECT id, ? FROM queries", (init_rank,))
            conn.execute("INSERT INTO f_rank(fragment_id, rank) SELECT id, ? FROM fragments", (init_rank,))

            last_delta = 0.0
            iters = 0
            for i in range(max_iter):
                iters = i + 1
                conn.execute(
                    """
                    CREATE TEMP TABLE f_rank_new AS
                    SELECT
                        f.id AS fragment_id,
                        ? + ? * COALESCE(SUM(qr.rank * e.weight / qd.degree), 0.0) AS rank
                    FROM fragments f
                    LEFT JOIN query_fragment_edges e ON e.fragment_id = f.id
                    LEFT JOIN q_rank qr ON qr.query_id = e.query_id
                    LEFT JOIN q_degree qd ON qd.query_id = e.query_id
                    GROUP BY f.id
                    """,
                    (base, damping),
                )

                conn.execute(
                    """
                    CREATE TEMP TABLE q_rank_new AS
                    SELECT
                        q.id AS query_id,
                        ? + ? * COALESCE(SUM(fr.rank * e.weight / fd.degree), 0.0) AS rank
                    FROM queries q
                    LEFT JOIN query_fragment_edges e ON e.query_id = q.id
                    LEFT JOIN f_rank_new fr ON fr.fragment_id = e.fragment_id
                    LEFT JOIN f_degree fd ON fd.fragment_id = e.fragment_id
                    GROUP BY q.id
                    """,
                    (base, damping),
                )

                delta_q = conn.execute(
                    """
                    SELECT COALESCE(SUM(ABS(n.rank - o.rank)), 0.0) AS d
                    FROM q_rank_new n
                    JOIN q_rank o ON o.query_id = n.query_id
                    """
                ).fetchone()["d"]
                delta_f = conn.execute(
                    """
                    SELECT COALESCE(SUM(ABS(n.rank - o.rank)), 0.0) AS d
                    FROM f_rank_new n
                    JOIN f_rank o ON o.fragment_id = n.fragment_id
                    """
                ).fetchone()["d"]
                last_delta = float(delta_q) + float(delta_f)

                conn.executescript(
                    """
                    DROP TABLE q_rank;
                    ALTER TABLE q_rank_new RENAME TO q_rank;
                    DROP TABLE f_rank;
                    ALTER TABLE f_rank_new RENAME TO f_rank;
                    """
                )
                if last_delta < tol:
                    break

            conn.execute(
                """
                INSERT INTO fragment_metrics(fragment_id, pagerank, query_support, raw_count, last_updated)
                SELECT
                    f.id,
                    COALESCE(fr.rank, 0.0) AS pagerank,
                    COALESCE(qs.query_support, 0) AS query_support,
                    f.raw_count,
                    CURRENT_TIMESTAMP
                FROM fragments f
                LEFT JOIN f_rank fr ON fr.fragment_id = f.id
                LEFT JOIN (
                    SELECT fragment_id, COUNT(*) AS query_support
                    FROM query_fragment_edges
                    GROUP BY fragment_id
                ) qs ON qs.fragment_id = f.id
                ON CONFLICT(fragment_id) DO UPDATE SET
                    pagerank = excluded.pagerank,
                    query_support = excluded.query_support,
                    raw_count = excluded.raw_count,
                    last_updated = excluded.last_updated
                """
            )
            conn.execute(
                """
                INSERT INTO query_metrics(query_id, pagerank, fragment_degree, raw_count, last_updated)
                SELECT
                    q.id,
                    COALESCE(qr.rank, 0.0) AS pagerank,
                    COALESCE(fd.fragment_degree, 0) AS fragment_degree,
                    q.raw_count,
                    CURRENT_TIMESTAMP
                FROM queries q
                LEFT JOIN q_rank qr ON qr.query_id = q.id
                LEFT JOIN (
                    SELECT query_id, COUNT(*) AS fragment_degree
                    FROM query_fragment_edges
                    GROUP BY query_id
                ) fd ON fd.query_id = q.id
                ON CONFLICT(query_id) DO UPDATE SET
                    pagerank = excluded.pagerank,
                    fragment_degree = excluded.fragment_degree,
                    raw_count = excluded.raw_count,
                    last_updated = excluded.last_updated
                """
            )

        return {"iterations": iters, "delta": last_delta}

    def top_fragments(
        self,
        limit: int = 50,
        kinds: Optional[Sequence[str]] = None,
        min_query_support: int = 2,
        alpha: float = 0.7,
    ) -> List[Dict[str, object]]:
        kinds = list(kinds or [])
        kind_filter = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_filter = f"AND f.kind IN ({placeholders})"

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    f.id,
                    f.kind,
                    f.canonical_sql,
                    fm.pagerank,
                    fm.query_support,
                    fm.raw_count
                FROM fragment_metrics fm
                JOIN fragments f ON f.id = fm.fragment_id
                WHERE fm.query_support >= ?
                {kind_filter}
                ORDER BY fm.pagerank DESC, fm.query_support DESC, fm.raw_count DESC
                """,
                (min_query_support, *kinds),
            ).fetchall()
        ranked: List[Dict[str, object]] = []
        for row in rows:
            combined_score = alpha * float(row["pagerank"]) + (1.0 - alpha) * math.log1p(
                int(row["query_support"])
            )
            ranked.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "canonical_sql": row["canonical_sql"],
                    "pagerank": float(row["pagerank"]),
                    "query_support": int(row["query_support"]),
                    "raw_count": int(row["raw_count"]),
                    "combined_score": combined_score,
                }
            )
        ranked.sort(key=lambda x: float(x["combined_score"]), reverse=True)
        return ranked[:limit]

    def export_graph_snapshot(
        self,
        json_path: str,
        dot_path: Optional[str] = None,
        max_fragments: int = 100,
        max_queries: int = 200,
    ) -> Dict[str, int]:
        with self._connect() as conn:
            fragment_rows = conn.execute(
                """
                SELECT f.id, f.kind, f.canonical_sql, fm.pagerank, fm.query_support
                FROM fragment_metrics fm
                JOIN fragments f ON f.id = fm.fragment_id
                ORDER BY fm.pagerank DESC, fm.query_support DESC
                LIMIT ?
                """,
                (max_fragments,),
            ).fetchall()
            fragment_ids = [row["id"] for row in fragment_rows]
            if not fragment_ids:
                payload = {"nodes": [], "edges": []}
                Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if dot_path:
                    Path(dot_path).write_text("graph SQLPatterns {}\n", encoding="utf-8")
                return {"nodes": 0, "edges": 0}

            placeholders = ",".join("?" for _ in fragment_ids)
            edge_rows = conn.execute(
                f"""
                SELECT query_id, fragment_id, SUM(weight) AS weight
                FROM query_fragment_edges
                WHERE fragment_id IN ({placeholders})
                GROUP BY query_id, fragment_id
                ORDER BY weight DESC
                LIMIT ?
                """,
                (*fragment_ids, max_queries * 10),
            ).fetchall()

            query_weight = Counter()
            for row in edge_rows:
                query_weight[row["query_id"]] += int(row["weight"])
            top_query_ids = [qid for qid, _ in query_weight.most_common(max_queries)]
            query_set = set(top_query_ids)

            query_rows: List[sqlite3.Row] = []
            if top_query_ids:
                q_placeholders = ",".join("?" for _ in top_query_ids)
                query_rows = conn.execute(
                    f"""
                    SELECT q.id, q.canonical_sql, qm.pagerank, q.raw_count
                    FROM queries q
                    LEFT JOIN query_metrics qm ON qm.query_id = q.id
                    WHERE q.id IN ({q_placeholders})
                    """,
                    top_query_ids,
                ).fetchall()

            nodes: List[Dict[str, object]] = []
            edges: List[Dict[str, object]] = []

            for row in fragment_rows:
                nodes.append(
                    {
                        "id": f"fragment:{row['id']}",
                        "type": "fragment",
                        "kind": row["kind"],
                        "pagerank": float(row["pagerank"]),
                        "query_support": int(row["query_support"]),
                        "sql": row["canonical_sql"],
                    }
                )

            for row in query_rows:
                nodes.append(
                    {
                        "id": f"query:{row['id']}",
                        "type": "query",
                        "pagerank": float(row["pagerank"] or 0.0),
                        "raw_count": int(row["raw_count"]),
                        "sql": row["canonical_sql"],
                    }
                )

            for row in edge_rows:
                if row["query_id"] in query_set:
                    edges.append(
                        {
                            "source": f"query:{row['query_id']}",
                            "target": f"fragment:{row['fragment_id']}",
                            "weight": int(row["weight"]),
                        }
                    )

        payload = {"nodes": nodes, "edges": edges}
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if dot_path:
            self._write_dot(dot_path, payload)
        return {"nodes": len(nodes), "edges": len(edges)}

    def _write_dot(self, dot_path: str, payload: Dict[str, object]) -> None:
        lines = ["graph SQLPatterns {"]
        lines.append('  graph [overlap=false, splines=true];')
        lines.append('  node [fontname="Helvetica"];')
        for node in payload["nodes"]:
            nid = str(node["id"]).replace(":", "_")
            label_prefix = "Q" if node["type"] == "query" else "F"
            label = f"{label_prefix}:{str(node['id']).split(':', 1)[1][:10]}"
            shape = "ellipse" if node["type"] == "query" else "box"
            lines.append(f'  "{nid}" [label="{label}", shape={shape}];')
        for edge in payload["edges"]:
            s = str(edge["source"]).replace(":", "_")
            t = str(edge["target"]).replace(":", "_")
            w = int(edge["weight"])
            lines.append(f'  "{s}" -- "{t}" [penwidth={1 + math.log1p(w):.3f}];')
        lines.append("}")
        Path(dot_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def summary(self) -> Dict[str, int]:
        with self._connect() as conn:
            return {
                "queries": int(conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]),
                "fragments": int(conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]),
                "edges": int(conn.execute("SELECT COUNT(*) FROM query_fragment_edges").fetchone()[0]),
            }


def iter_sql_from_file(path: str, sql_key: str = "sql") -> Iterator[str]:
    p = Path(path)
    suffix = p.suffix.lower()
    with p.open("r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".ndjson"}:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sql = str(obj.get(sql_key, "")).strip()
                if sql:
                    yield sql
            return

        for line in f:
            line = line.strip()
            if line:
                yield line
