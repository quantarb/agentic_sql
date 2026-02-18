from typing import Any, Dict, List


def format_tool_outputs(
    db_id: str,
    schema_hint: str,
    retrieved_reason: List[Dict[str, Any]],
    global_candidates: List[Dict[str, Any]],
    pre_ctx: List[Dict[str, Any]],
    qa_ctx: List[Dict[str, Any]],
    max_chars: int = 1200,
) -> str:
    lines: List[str] = [f"db_id={db_id}"]
    lines.append(
        "schema_hint_available=yes"
        if schema_hint and schema_hint != "(none yet — you haven't verified any schema for this DB_ID)"
        else "schema_hint_available=no"
    )
    lines.append(
        "reasoning_lessons_top="
        + (", ".join(str(x.get("title", "")).strip() for x in retrieved_reason[:3]) if retrieved_reason else "(none)")
    )
    lines.append(
        "global_rules_top="
        + (", ".join(str(x.get("title", "")).strip() for x in global_candidates[:3]) if global_candidates else "(none)")
    )
    lines.append(
        "context_hits_top="
        + (", ".join(f"{x.get('layer')}::{x.get('title')}" for x in pre_ctx[:4]) if pre_ctx else "(none)")
    )
    lines.append(
        "related_qa_top="
        + (", ".join(str(x.get("title", "")).strip() for x in qa_ctx[:4]) if qa_ctx else "(none)")
    )
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "... [truncated]"
    return out


def format_related_qa(qa_ctx: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    if not qa_ctx:
        return "(none)"
    chunks: List[str] = []
    used = 0
    for i, row in enumerate(qa_ctx, 1):
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 420:
            content = content[:420] + "..."
        chunk = f"[Related QA {i} | score={row.get('score', 0.0):.3f}]\n{content}\n"
        if used + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n".join(chunks).strip() if chunks else "(none)"


def merge_ranked_hits(*hit_lists: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    for hits in hit_lists:
        for h in hits:
            hid = int(h.get("id", -1))
            if hid < 0:
                continue
            prev = by_id.get(hid)
            if prev is None or float(h.get("score", 0.0)) > float(prev.get("score", 0.0)):
                by_id[hid] = h
    ranked = sorted(by_id.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ranked[: max(0, int(limit))]


def format_context_chunks(rows: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    if not rows:
        return "(none)"
    chunks: List[str] = []
    used = 0
    for j, row in enumerate(rows, 1):
        txt = (row.get("content") or "").strip()
        if len(txt) > 700:
            txt = txt[:700] + "..."
        chunk = f"[Context {j} | layer={row.get('layer')} score={row.get('score', 0.0):.3f}]\n{txt}\n"
        if used + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n".join(chunks).strip() if chunks else "(none)"
