from typing import Any, Dict, List

from agentic_sql.config import ExperimentConfig
from agentic_sql.llm import TextToSQLLLM
from agentic_sql.memory import GlobalBusinessRulesMemory, ReasoningMemory


def lesson_evidence(meta: Dict[str, Any]) -> int:
    try:
        return max(1, int(meta.get("evidence_count", 1)))
    except Exception:
        return 1


def lesson_confidence(meta: Dict[str, Any]) -> float:
    try:
        return float(meta.get("confidence", 0.0))
    except Exception:
        return 0.0


def _is_retrievable_candidate(cand: Dict[str, Any], include_tentative: bool) -> bool:
    meta = cand.get("meta") or {}
    if include_tentative:
        return True
    return str(meta.get("status", "active")) != "tentative"


def prune_expired_tentative_lessons(
    reason_memory: ReasoningMemory,
    current_step: int,
    ttl_steps: int,
) -> int:
    if ttl_steps <= 0 or reason_memory.size <= 0:
        return 0
    drop_ids: List[int] = []
    for idx, meta in enumerate(reason_memory.meta):
        if str(meta.get("status", "active")) != "tentative":
            continue
        last_seen = int(meta.get("last_seen_step", 0) or 0)
        if last_seen <= 0:
            continue
        if (current_step - last_seen) > ttl_steps:
            drop_ids.append(idx)
    return reason_memory.remove_lessons(drop_ids)


def retrieve_reasoning_lessons(
    llm: TextToSQLLLM,
    reason_memory: ReasoningMemory,
    db_id: str,
    question: str,
    initial_k: int = 20,
    top_k: int = 3,
    include_tentative: bool = False,
) -> List[Dict[str, Any]]:
    candidates = reason_memory.retrieve_candidates(retrieval_key=f"{db_id}\n{question}", k=initial_k)
    if not candidates:
        return []
    candidates = [c for c in candidates if _is_retrievable_candidate(c, include_tentative=include_tentative)]
    if not candidates:
        return []

    reranked_ids = llm.rerank_lesson_ids(question=question, db_id=db_id, candidates=candidates, top_k=top_k)
    if not reranked_ids:
        return candidates[:top_k]
    return ReasoningMemory.select_by_ids(candidates, reranked_ids)


def consolidate_memory(
    llm: TextToSQLLLM,
    reason_memory: ReasoningMemory,
    global_memory: GlobalBusinessRulesMemory,
    step_idx: int,
    every_n_steps: int,
    consolidation_k: int,
    verbose: bool = True,
) -> Dict[str, int]:
    stats = {"merged_reason": 0, "merged_global": 0}
    if step_idx <= 0 or (step_idx % max(1, every_n_steps)) != 0:
        return stats

    def _consolidate_one(memory: Any, memory_name: str) -> int:
        merged_count = 0
        for i, lesson in enumerate(list(memory.lessons)):
            base_meta = memory.meta[i]
            if str(base_meta.get("status", "active")) == "tentative":
                continue
            key = memory.meta[i].get("retrieval_key", "")
            cands = memory.retrieve_candidates(key, k=consolidation_k + 1)
            cands = [c for c in cands if c["id"] != i][:consolidation_k]
            if not cands:
                continue
            decision = llm.consolidate_candidate_lessons(new_lesson=lesson, similar_lessons=cands)
            merge_ids = [x for x in decision.get("merge_ids", []) if x != i]
            if decision.get("action") != "merge" or not merge_ids:
                continue

            merged_from = [i] + [x for x in merge_ids if 0 <= x < len(memory.meta)]
            merged_meta_list = [memory.meta[x] for x in merged_from]
            merged_meta = {
                "retrieval_key": key,
                "consolidated": True,
                "memory": memory_name,
                "status": "active" if any(bool(m.get("verified", False)) for m in merged_meta_list) else "tentative",
                "verified": any(bool(m.get("verified", False)) for m in merged_meta_list),
                "evidence_count": sum(lesson_evidence(m) for m in merged_meta_list),
                "confidence": max((lesson_confidence(m) for m in merged_meta_list), default=0.0),
                "last_seen_step": max((int(m.get("last_seen_step", 0) or 0) for m in merged_meta_list), default=0),
                "last_verified_step": max(
                    (int(m.get("last_verified_step", 0) or 0) for m in merged_meta_list),
                    default=0,
                ),
            }
            if memory_name == "global":
                merged_meta["scope"] = "global"
                merged_meta["owner_scope"] = "global"
            else:
                merged_meta["scope"] = "db_specific"
                merged_meta["owner_scope"] = "db_specific"

            memory.replace_lessons(
                ids_to_replace=[i] + merge_ids,
                merged_text=decision.get("merged_text", lesson),
                merged_meta=merged_meta,
            )
            merged_count += 1
            break
        return merged_count

    stats["merged_reason"] = _consolidate_one(reason_memory, "reason")
    stats["merged_global"] = _consolidate_one(global_memory, "global")

    if verbose and (stats["merged_reason"] or stats["merged_global"]):
        print(
            f"  [consolidation] merged_reason={stats['merged_reason']} "
            f"merged_global={stats['merged_global']}"
        )

    return stats


def store_with_knowledge_manager(
    llm: TextToSQLLLM,
    reason_memory: ReasoningMemory,
    global_memory: GlobalBusinessRulesMemory,
    lesson_text: str,
    db_id: str,
    question: str,
    row_idx: int,
    sim_before: float,
    sim_after: float,
    consolidation_k: int,
    cfg: ExperimentConfig,
    verified: bool = True,
    failure_reason: str = "",
) -> str:
    similar = reason_memory.retrieve_candidates(f"{db_id}\n{question}", k=consolidation_k)
    decision = llm.consolidate_candidate_lessons(new_lesson=lesson_text, similar_lessons=similar)

    merged_text = decision.get("merged_text", lesson_text)
    merge_ids = [int(x) for x in decision.get("merge_ids", []) if isinstance(x, int) or str(x).isdigit()]
    merge_ids = [x for x in merge_ids if 0 <= x < len(reason_memory.meta)]
    scope = decision.get("scope", "db_specific")
    if not verified:
        scope = "db_specific"

    merged_metas = [reason_memory.meta[x] for x in merge_ids]
    evidence_count = 1 + sum(lesson_evidence(m) for m in merged_metas)
    sim_lift = max(0.0, float(sim_after) - float(sim_before))
    confidence = 0.25 if not verified else min(0.99, 0.55 + 2.0 * sim_lift)
    status = "active" if verified else "tentative"
    owner_scope = "global" if scope == "global" else "db_specific"

    if scope == "global" and (
        confidence < float(cfg.min_global_confidence) or evidence_count < int(cfg.min_global_evidence)
    ):
        scope = "db_specific"
        owner_scope = "db_specific"

    meta = {
        "db_id": db_id,
        "row_idx": row_idx,
        "retrieval_key": f"{db_id}\n{question}",
        "sim_before": float(sim_before),
        "sim_after": float(sim_after),
        "sim_lift": sim_lift,
        "scope": scope,
        "owner_scope": owner_scope,
        "verified": bool(verified),
        "status": status,
        "confidence": confidence,
        "evidence_count": evidence_count,
        "last_seen_step": int(row_idx) + 1,
        "last_verified_step": (int(row_idx) + 1) if verified else 0,
        "ttl_steps": 0 if verified else int(cfg.tentative_ttl_steps),
        "failure_reason": failure_reason,
    }

    if scope == "global":
        global_memory.add_rule(merged_text, meta=meta)
        return "global"

    if decision.get("action") == "merge" and merge_ids:
        reason_memory.replace_lessons(ids_to_replace=merge_ids, merged_text=merged_text, merged_meta=meta)
    else:
        reason_memory.add_lesson(merged_text, meta=meta)

    return "reason"
