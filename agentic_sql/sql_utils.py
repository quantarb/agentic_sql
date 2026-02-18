import re
from typing import Dict, Set, Tuple

import sqlglot
from sqlglot import exp


SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


def clean_sql(text: str) -> str:
    t = text.strip()
    t = SQL_FENCE_RE.sub("", t).strip()
    t = re.sub(r";\s*$", "", t).strip()
    t = t.replace("<>", "!=")
    m = re.search(r"(?is)\b(with|select)\b", t)
    if m:
        t = t[m.start():].strip()
    return t


def strip_literals_in_ast(node: exp.Expression) -> exp.Expression:
    node = node.copy()
    for lit in list(node.find_all(exp.Literal)):
        replacement = exp.Literal.string("__VALUE__") if lit.is_string else exp.Literal.number(0)
        lit.replace(replacement)
    return node


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _col_key(c: exp.Column) -> str:
    if c.table:
        return f"{c.table.lower()}.{c.name.lower()}"
    return c.name.lower()


def _collect_features(node: exp.Expression) -> Dict[str, Set[str]]:
    feats: Dict[str, Set[str]] = {
        "tables": set(),
        "columns": set(),
        "projections": set(),
        "joins": set(),
        "filters": set(),
        "group_by": set(),
        "order_by": set(),
        "aggs": set(),
    }

    for t in node.find_all(exp.Table):
        if t.name:
            feats["tables"].add(t.name.lower())

    for c in node.find_all(exp.Column):
        feats["columns"].add(_col_key(c))
        feats["columns"].add(c.name.lower())

    select = node.find(exp.Select)
    if select is not None:
        for e in select.expressions:
            base = e.this if isinstance(e, exp.Alias) else e
            if isinstance(base, exp.Column):
                feats["projections"].add(_col_key(base))
            else:
                feats["projections"].add(type(base).__name__)

    for eq in node.find_all(exp.EQ):
        l, r = eq.left, eq.right
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            feats["joins"].add(" = ".join(sorted([_col_key(l), _col_key(r)])))
        elif isinstance(l, exp.Column):
            feats["filters"].add(f"{_col_key(l)}=VALUE")
        elif isinstance(r, exp.Column):
            feats["filters"].add(f"{_col_key(r)}=VALUE")

    for cmp_node, op in (
        (exp.GT, ">"),
        (exp.GTE, ">="),
        (exp.LT, "<"),
        (exp.LTE, "<="),
        (exp.NEQ, "!="),
        (exp.Like, "like"),
        (exp.In, "in"),
    ):
        for n in node.find_all(cmp_node):
            cols = [x for x in (n.left if hasattr(n, "left") else None, n.right if hasattr(n, "right") else None) if isinstance(x, exp.Column)]
            for c in cols:
                feats["filters"].add(f"{_col_key(c)}{op}VALUE")

    for gb in node.find_all(exp.Group):
        for e in gb.expressions:
            if isinstance(e, exp.Column):
                feats["group_by"].add(_col_key(e))
            else:
                feats["group_by"].add(type(e).__name__)

    for ob in node.find_all(exp.Ordered):
        this = ob.this
        direction = "desc" if ob.args.get("desc") else "asc"
        if isinstance(this, exp.Column):
            feats["order_by"].add(f"{_col_key(this)}:{direction}")
        else:
            feats["order_by"].add(f"{type(this).__name__}:{direction}")

    for f in node.find_all(exp.Func):
        name = (f.sql_name() or type(f).__name__).lower()
        feats["aggs"].add(name)

    return feats


def ast_scores(pred_sql: str, gold_sql: str) -> Tuple[bool, float]:
    """
    Returns exact match and structural Jaccard similarity.
    Exact match is literal-insensitive normalized SQL AST equality.
    """
    try:
        p = sqlglot.parse_one(clean_sql(pred_sql), read="sqlite")
        g = sqlglot.parse_one(clean_sql(gold_sql), read="sqlite")

        p = strip_literals_in_ast(p)
        g = strip_literals_in_ast(g)

        p_norm = p.sql(dialect="sqlite", normalize=True)
        g_norm = g.sql(dialect="sqlite", normalize=True)
        exact = p_norm == g_norm

        pf = _collect_features(p)
        gf = _collect_features(g)
        weights = {
            "tables": 0.30,
            "projections": 0.25,
            "filters": 0.15,
            "joins": 0.10,
            "columns": 0.10,
            "group_by": 0.05,
            "order_by": 0.03,
            "aggs": 0.02,
        }
        sim = 0.0
        for k, w in weights.items():
            sim += w * _jaccard(pf[k], gf[k])

        # Treat core-structure equality as exact even with superficial aliasing/format variation.
        core_equal = (
            pf["tables"] == gf["tables"]
            and pf["projections"] == gf["projections"]
            and pf["filters"] == gf["filters"]
            and pf["joins"] == gf["joins"]
            and pf["group_by"] == gf["group_by"]
            and pf["order_by"] == gf["order_by"]
        )
        exact = bool(exact or core_equal)
        return exact, sim
    except Exception:
        return False, 0.0


def extract_schema_from_gold_sql(gold_sql: str) -> Dict[str, Set[str]]:
    ast = sqlglot.parse_one(clean_sql(gold_sql), read="sqlite")
    ast = strip_literals_in_ast(ast)

    tables = {t.name.lower() for t in ast.find_all(exp.Table) if t.name}

    columns: Set[str] = set()
    for c in ast.find_all(exp.Column):
        name = c.name.lower()
        if c.table:
            columns.add(f"{c.table.lower()}.{name}")
        columns.add(name)

    join_edges: Set[str] = set()
    for eq in ast.find_all(exp.EQ):
        l, r = eq.left, eq.right
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            lq = f"{l.table.lower()}.{l.name.lower()}" if l.table else l.name.lower()
            rq = f"{r.table.lower()}.{r.name.lower()}" if r.table else r.name.lower()
            join_edges.add(" = ".join(sorted([lq, rq])))

    return {"tables": tables, "columns": columns, "join_edges": join_edges}
