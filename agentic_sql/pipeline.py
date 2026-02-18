from collections import defaultdict
import os
from typing import Any, Dict, List, Tuple

from agentic_sql.config import ExperimentConfig
from agentic_sql import pipeline_formatting
from agentic_sql import pipeline_memory_ops
from agentic_sql.kb_store import KnowledgeBaseStore
from agentic_sql.llm import TextToSQLLLM
from agentic_sql.memory import GlobalBusinessRulesMemory, ReasoningMemory, SchemaMemory
from agentic_sql.observability import PhoenixObservability
from agentic_sql.sql_utils import ast_scores, extract_schema_from_gold_sql


def run_dual_memory_experiment(
    examples: List[Dict[str, Any]],
    llm: TextToSQLLLM,
    schema_memory: SchemaMemory,
    reason_memory: ReasoningMemory,
    global_memory: GlobalBusinessRulesMemory | None = None,
    context_builder: Any | None = None,
    kb_store: KnowledgeBaseStore | None = None,
    cfg: ExperimentConfig | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cfg = cfg or ExperimentConfig()
    global_memory = global_memory or GlobalBusinessRulesMemory(llm.embed_texts)

    stats: Dict[str, Any] = {
        "n": 0,
        "pass_first": 0,
        "fail_first": 0,
        "pass_retry": 0,
        "pass_final": 0,
        "exact_first": 0,
        "exact_retry": 0,
        "exact_final": 0,
        "sim1_sum": 0.0,
        "sim2_sum": 0.0,
        "sim2_count": 0,
        "sim_lift_sum": 0.0,
        "sim_lift_count": 0,
        "high_sim_non_exact_final": 0,
        "lessons_written": 0,
        "reason_lessons_stored": 0,
        "global_rules_stored": 0,
        "tentative_lessons_stored": 0,
        "expired_tentative_pruned": 0,
        "schema_updates": 0,
        "consolidation_merges": 0,
        "ab_samples": 0,
        "ab_with_memory_better": 0,
        "ab_without_memory_better": 0,
        "ab_ties": 0,
        "ab_pass_with_memory": 0,
        "ab_pass_without_memory": 0,
        "planner_runs": 0,
    }
    history: List[Dict[str, Any]] = []
    db_totals: Dict[str, int] = defaultdict(int)
    db_final_pass: Dict[str, int] = defaultdict(int)
    db_exact_pass: Dict[str, int] = defaultdict(int)
    run_id = kb_store.start_run(config=cfg.__dict__) if kb_store else None
    phoenix = PhoenixObservability(
        enabled=cfg.phoenix_enabled,
        project_name=cfg.phoenix_project_name,
        endpoint=cfg.phoenix_endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", ""),
    )
    stats["phoenix_enabled"] = bool(phoenix.enabled)
    stats["phoenix_setup_error"] = phoenix.setup_error

    def _log_observability(step_idx: int, event_type: str, payload: Dict[str, Any]) -> None:
        if run_id is not None:
            phoenix.log_event(run_id=run_id, step_idx=step_idx, event_type=event_type, payload=payload)
            if kb_store is not None:
                kb_store.write_observability_event(
                    run_id=run_id,
                    step_idx=step_idx,
                    event_type=event_type,
                    payload=payload,
                )

    def _record_llm_call(step_idx: int, stage: str, expected_operation: str | None = None) -> Dict[str, Any]:
        meta = llm.consume_last_call_meta()
        if not meta:
            return {}
        if expected_operation and str(meta.get("operation", "")) != expected_operation:
            return {}
        stats["llm_calls"] = int(stats.get("llm_calls", 0)) + 1
        stats["llm_input_tokens"] = int(stats.get("llm_input_tokens", 0)) + int(meta.get("input_tokens", 0) or 0)
        stats["llm_output_tokens"] = int(stats.get("llm_output_tokens", 0)) + int(meta.get("output_tokens", 0) or 0)
        stats["llm_total_tokens"] = int(stats.get("llm_total_tokens", 0)) + int(meta.get("total_tokens", 0) or 0)
        stats["llm_estimated_cost_usd"] = float(stats.get("llm_estimated_cost_usd", 0.0)) + float(
            meta.get("estimated_cost_usd", 0.0) or 0.0
        )
        stats["llm_latency_ms_sum"] = float(stats.get("llm_latency_ms_sum", 0.0)) + float(meta.get("latency_ms", 0.0) or 0.0)
        _log_observability(step_idx=step_idx, event_type="llm_call", payload={"stage": stage, **meta})
        return meta

    def _clip(text: str, max_chars: int) -> str:
        t = (text or "").strip()
        if len(t) <= max_chars:
            return t
        return t[:max_chars] + "... [truncated]"

    n_steps = min(cfg.n_steps, len(examples))
    for i in range(n_steps):
        pruned = pipeline_memory_ops.prune_expired_tentative_lessons(
            reason_memory=reason_memory,
            current_step=i + 1,
            ttl_steps=cfg.tentative_ttl_steps,
        )
        if pruned:
            stats["expired_tentative_pruned"] += int(pruned)
            if verbose:
                print(f"[{i+1}] pruned_expired_tentative={pruned}")

        ex = examples[i]
        question = ex["question"]
        db_id = ex["db_id"]
        gold = ex["query"]

        # If no schema is known for this DB yet, bootstrap from AST identifiers.
        # This avoids cold-start failure loops where schema never updates because nothing passes.
        if not schema_memory.has_schema(db_id):
            seeded = schema_memory.update_from_sql_ast(db_id=db_id, sql_text=gold, source="gold_ast_fallback")
            if seeded:
                stats["schema_updates"] += 1
                if verbose:
                    print(f"[{i+1}] schema-bootstrapped from AST for DB_ID={db_id}")

        schema_hint = schema_memory.format_hint(db_id)

        retrieved_reason = pipeline_memory_ops.retrieve_reasoning_lessons(
            llm=llm,
            reason_memory=reason_memory,
            db_id=db_id,
            question=question,
            initial_k=cfg.initial_retrieval_k,
            top_k=cfg.retrieval_k,
            include_tentative=cfg.include_tentative_in_retrieval,
        )
        rerank_meta = _record_llm_call(step_idx=i, stage="reason_rerank", expected_operation="rerank_lesson_ids")
        reason_text = ReasoningMemory.format(retrieved_reason, label="Reasoning Lesson")

        global_candidates = global_memory.retrieve(question=question, k=cfg.global_rule_k)
        global_text = GlobalBusinessRulesMemory.format(global_candidates, label="Global Business Rule")

        preprocessed_context_text = "(none)"
        pre_ctx: List[Dict[str, Any]] = []
        qa_ctx: List[Dict[str, Any]] = []
        if context_builder is not None:
            try:
                overall_ctx = context_builder.retrieve(
                    question=question,
                    embed_fn=llm.embed_texts,
                    top_k=cfg.context_top_k,
                    db_id=db_id,
                    runtime_first_k=cfg.runtime_context_first_k,
                    runtime_boost=cfg.runtime_context_boost,
                    runtime_max_age_hours=cfg.runtime_context_max_age_hours,
                )
                layer_hits: List[Dict[str, Any]] = []
                for layer in (
                    "runtime_context",
                    "table_usage",
                    "human_annotations",
                    "institutional_knowledge",
                    "codex_enrichment",
                    "memory",
                ):
                    layer_hits.extend(
                        context_builder.retrieve(
                            question=question,
                            embed_fn=llm.embed_texts,
                            top_k=cfg.context_per_layer_k,
                            db_id=db_id,
                            layers=[layer],
                            runtime_first_k=cfg.runtime_context_first_k if layer == "runtime_context" else 0,
                            runtime_boost=cfg.runtime_context_boost if layer == "runtime_context" else 0.0,
                            runtime_max_age_hours=cfg.runtime_context_max_age_hours if layer == "runtime_context" else 0,
                        )
                    )
                pre_ctx = pipeline_formatting.merge_ranked_hits(overall_ctx, layer_hits, limit=cfg.context_top_k)
            except Exception:
                pre_ctx = []
            preprocessed_context_text = pipeline_formatting.format_context_chunks(
                pre_ctx, max_chars=max(600, cfg.context_max_chars)
            )
            if cfg.qa_top_k > 0:
                try:
                    qa_ctx = context_builder.retrieve(
                        question=question,
                        embed_fn=llm.embed_texts,
                        top_k=cfg.qa_top_k,
                        db_id=db_id,
                        layers=["memory"],
                    )
                except Exception:
                    qa_ctx = []
        related_qa_text = pipeline_formatting.format_related_qa(qa_ctx, max_chars=max(300, cfg.qa_max_chars))
        tool_outputs_text = pipeline_formatting.format_tool_outputs(
            db_id=db_id,
            schema_hint=schema_hint,
            retrieved_reason=retrieved_reason,
            global_candidates=global_candidates,
            pre_ctx=pre_ctx,
            qa_ctx=qa_ctx,
            max_chars=max(300, cfg.planner_include_context_chars),
        )
        plan_obj = llm.plan_sql_approach(
            question=question,
            db_id=db_id,
            schema_hint=schema_hint,
            reasoning_lessons=f"[DB-SPECIFIC]\n{reason_text}\n\n[GLOBAL-RULES]\n{global_text}",
            tool_outputs=tool_outputs_text,
        )
        planner_meta = _record_llm_call(step_idx=i, stage="planner", expected_operation="plan_sql_approach")
        planner_text = llm.format_plan(plan_obj, max_chars=max(300, cfg.planner_include_context_chars))
        stats["planner_runs"] += 1
        _log_observability(
            step_idx=i,
            event_type="retrieval",
            payload={
                "db_id": db_id,
                "reason_hits": [
                    {"id": int(x["id"]), "title": x.get("title", ""), "score": float(x.get("score", 0.0))}
                    for x in retrieved_reason
                ],
                "global_hits": [
                    {"id": int(x["id"]), "title": x.get("title", ""), "score": float(x.get("score", 0.0))}
                    for x in global_candidates
                ],
                "context_hits": [
                    {
                        "id": int(x["id"]),
                        "layer": str(x.get("layer", "")),
                        "title": str(x.get("title", "")),
                        "score": float(x.get("score", 0.0)),
                    }
                    for x in (pre_ctx if context_builder is not None else [])
                ][: cfg.context_top_k],
                "qa_hits": [
                    {
                        "id": int(x["id"]),
                        "title": str(x.get("title", "")),
                        "score": float(x.get("score", 0.0)),
                    }
                    for x in qa_ctx
                ][: cfg.qa_top_k],
                "rerank_meta": rerank_meta,
                "planner_meta": planner_meta,
            },
        )

        pred1 = llm.generate_sql(
            question,
            db_id,
            schema_hint,
            f"[DB-SPECIFIC]\n{reason_text}\n\n"
            f"[GLOBAL-RULES]\n{global_text}\n\n"
            f"[RELATED-QA]\n{related_qa_text}\n\n"
            f"[TOOL-OUTPUTS]\n{tool_outputs_text}\n\n"
            f"[PLANNER]\n{planner_text}\n\n"
            f"[PREPROCESSED-CONTEXT]\n{preprocessed_context_text}",
        )
        pred1_meta = _record_llm_call(step_idx=i, stage="pred1_generate", expected_operation="generate_sql")
        if verbose and cfg.debug_trace:
            print("=" * 80)
            print(f"[{i+1}] DEBUG INPUT")
            print(f"DB_ID: {db_id}")
            print("QUESTION:")
            print(_clip(question, cfg.debug_context_chars))
            print("\nSCHEMA_HINT:")
            print(_clip(schema_hint, cfg.debug_context_chars))
            print("\nDB-SPECIFIC LESSONS:")
            print(_clip(reason_text, cfg.debug_context_chars))
            print("\nGLOBAL RULES:")
            print(_clip(global_text, cfg.debug_context_chars))
            print("\nRELATED-QA:")
            print(_clip(related_qa_text, cfg.debug_context_chars))
            print("\nTOOL OUTPUTS:")
            print(_clip(tool_outputs_text, cfg.debug_context_chars))
            print("\nPLANNER:")
            print(_clip(planner_text, cfg.debug_context_chars))
            print("\nPREPROCESSED CONTEXT:")
            print(_clip(preprocessed_context_text, cfg.debug_context_chars))
        exact1, sim1 = ast_scores(pred1, gold)
        pass1 = exact1 or (sim1 >= cfg.pass_sim_threshold)

        # If generated SQL references tables not in known schema, refresh schema from parsed gold SQL.
        try:
            pred_tables = extract_schema_from_gold_sql(pred1)["tables"]
        except Exception:
            pred_tables = set()
        known_tables = set(schema_memory.get(db_id)["tables"].keys())
        unknown_pred_tables = sorted([t for t in pred_tables if t not in known_tables])
        if unknown_pred_tables:
            seeded = schema_memory.update_from_sql_ast(
                db_id=db_id,
                sql_text=gold,
                source="gold_ast_on_unknown_table",
            )
            if seeded:
                stats["schema_updates"] += 1
                if verbose:
                    print(
                        f"[{i+1}] schema-refreshed from AST (unknown predicted tables: {unknown_pred_tables[:5]})"
                    )

        stats["n"] += 1
        stats["sim1_sum"] += float(sim1)
        if exact1:
            stats["exact_first"] += 1
        db_totals[db_id] += 1

        rec: Dict[str, Any] = {
            "index": i,
            "db_id": db_id,
            "question": question,
            "gold": gold,
            "pred1": pred1,
            "exact1": exact1,
            "sim1": sim1,
            "pass1": pass1,
            "unknown_pred_tables": unknown_pred_tables,
            "obs_pred1": pred1_meta,
            "planner_text": planner_text,
            "related_qa_count": len(qa_ctx),
        }

        if pass1:
            stats["pass_first"] += 1
            stats["pass_final"] += 1
            if exact1:
                stats["exact_final"] += 1
                db_exact_pass[db_id] += 1
            elif sim1 >= cfg.pass_sim_threshold:
                stats["high_sim_non_exact_final"] += 1
            db_final_pass[db_id] += 1
            schema_memory.update_from_success_sql(
                db_id=db_id,
                question=question,
                successful_sql=gold,
                annotate_fn=llm.annotate_schema_from_success,
            )
            stats["schema_updates"] += 1

            if verbose:
                print(
                    f"[{i+1}] PASS-first sim={sim1:.3f} "
                    f"reason_lessons={reason_memory.size} global_rules={global_memory.size} "
                    f"schema_tables={len(schema_memory.get(db_id)['tables'])}"
                )

            cstats = pipeline_memory_ops.consolidate_memory(
                llm=llm,
                reason_memory=reason_memory,
                global_memory=global_memory,
                step_idx=i + 1,
                every_n_steps=cfg.consolidation_every_n,
                consolidation_k=cfg.consolidation_k,
                verbose=verbose,
            )
            stats["consolidation_merges"] += cstats["merged_reason"] + cstats["merged_global"]
            if kb_store and run_id is not None:
                kb_store.write_step_event(run_id=run_id, rec=rec, stored_bucket="")
                kb_store.persist_snapshot(
                    run_id=run_id,
                    reason_memory=reason_memory,
                    global_memory=global_memory,
                    schema_memory=schema_memory,
                )
            _log_observability(
                step_idx=i,
                event_type="outcome",
                payload={
                    "db_id": db_id,
                    "pass1": bool(pass1),
                    "pass2": False,
                    "exact1": bool(exact1),
                    "exact2": False,
                    "sim1": float(sim1),
                    "sim2": None,
                    "pass_final": True,
                    "execution_outcome": "ast_eval_only",
                },
            )
            history.append(rec)
            continue

        stats["fail_first"] += 1
        if verbose:
            print(f"[{i+1}] FAIL-first sim={sim1:.3f}")
            print("Q:", question)
            print("PRED1:", pred1)
            print("GOLD :", gold)

        critic_trace = llm.synthetic_traceback(
            question=question,
            db_id=db_id,
            pred_sql=pred1,
            gold_sql=gold,
            schema_hint=schema_hint,
        )
        critic_meta = _record_llm_call(
            step_idx=i,
            stage="critic_traceback",
            expected_operation="synthetic_traceback",
        )

        lesson_raw = llm.write_lesson(
            question=question,
            db_id=db_id,
            pred_sql=pred1,
            gold_sql=gold,
            synthetic_traceback=critic_trace,
        )
        lesson_meta = _record_llm_call(step_idx=i, stage="lesson_write", expected_operation="write_lesson")
        lesson_text = llm.normalize_lesson(lesson_raw)
        stats["lessons_written"] += 1
        if verbose and cfg.debug_trace:
            print("\nCRITIC TRACEBACK:")
            print(_clip(critic_trace, cfg.debug_context_chars))
            print("\nLESSON WRITTEN:")
            print(_clip(lesson_text, cfg.debug_context_chars))

        retry_reason_text = (
            f"[NEW LESSON]\n{lesson_text}\n\n"
            f"[SYNTHETIC_TRACEBACK]\n{critic_trace}\n\n"
            f"[PRIOR DB-SPECIFIC]\n{reason_text}\n\n"
            f"[GLOBAL-RULES]\n{global_text}\n\n"
            f"[RELATED-QA]\n{related_qa_text}\n\n"
            f"[TOOL-OUTPUTS]\n{tool_outputs_text}\n\n"
            f"[PLANNER]\n{planner_text}\n\n"
            f"[PREPROCESSED-CONTEXT]\n{preprocessed_context_text}"
        )
        pred2 = llm.generate_sql(question, db_id, schema_hint, retry_reason_text)
        pred2_meta = _record_llm_call(step_idx=i, stage="pred2_generate", expected_operation="generate_sql")
        if verbose and cfg.debug_trace:
            print("\nRETRY CONTEXT:")
            print(_clip(retry_reason_text, cfg.debug_context_chars))
            print("\nPRED2:")
            print(_clip(pred2, cfg.debug_context_chars))

        exact2, sim2 = ast_scores(pred2, gold)
        pass2 = exact2 or (sim2 >= cfg.pass_sim_threshold)
        stats["sim2_sum"] += float(sim2)
        stats["sim2_count"] += 1
        stats["sim_lift_sum"] += float(sim2 - sim1)
        stats["sim_lift_count"] += 1

        rec.update(
            {
                "critic_trace": critic_trace,
                "lesson": lesson_text,
                "pred2": pred2,
                "exact2": exact2,
                "sim2": sim2,
                "pass2": pass2,
                "obs_critic": critic_meta,
                "obs_lesson": lesson_meta,
                "obs_pred2": pred2_meta,
            }
        )

        stored_bucket = ""
        if pass2:
            stats["pass_retry"] += 1
            stats["pass_final"] += 1
            if exact2:
                stats["exact_retry"] += 1
                stats["exact_final"] += 1
                db_exact_pass[db_id] += 1
            elif sim2 >= cfg.pass_sim_threshold:
                stats["high_sim_non_exact_final"] += 1
            db_final_pass[db_id] += 1
            schema_memory.update_from_success_sql(
                db_id=db_id,
                question=question,
                successful_sql=gold,
                annotate_fn=llm.annotate_schema_from_success,
            )
            stats["schema_updates"] += 1

            bucket = pipeline_memory_ops.store_with_knowledge_manager(
                llm=llm,
                reason_memory=reason_memory,
                global_memory=global_memory,
                lesson_text=lesson_text,
                db_id=db_id,
                question=question,
                row_idx=i,
                sim_before=sim1,
                sim_after=sim2,
                consolidation_k=cfg.consolidation_k,
                cfg=cfg,
            )
            if bucket == "global":
                stats["global_rules_stored"] += 1
            else:
                stats["reason_lessons_stored"] += 1
            stored_bucket = bucket

            if verbose:
                print(
                    f"  PASS-retry sim={sim2:.3f} "
                    f"(stored={bucket}; updated schema for {db_id})"
                )
                print(
                    f"     schema_tables={len(schema_memory.get(db_id)['tables'])} "
                    f"schema_cols={len(schema_memory.get(db_id)['columns'])} "
                    f"reason_lessons={reason_memory.size} global_rules={global_memory.size}"
                )
        elif verbose:
            print("  FAIL-retry sim={:.3f} (will store tentative lesson)".format(sim2))
        else:
            pass

        if not pass2:
            # Keep a tentative lesson even on failure to improve next-attempt table choices.
            bucket = pipeline_memory_ops.store_with_knowledge_manager(
                llm=llm,
                reason_memory=reason_memory,
                global_memory=global_memory,
                lesson_text=lesson_text,
                db_id=db_id,
                question=question,
                row_idx=i,
                sim_before=sim1,
                sim_after=sim2,
                consolidation_k=cfg.consolidation_k,
                cfg=cfg,
                verified=False,
                failure_reason="retry_failed",
            )
            if bucket == "reason":
                stats["tentative_lessons_stored"] += 1
                stored_bucket = "reason_tentative"
                if verbose:
                    print("  stored tentative lesson for future retrieval")

        if cfg.memory_ab_every_n > 0 and ((i + 1) % cfg.memory_ab_every_n == 0):
            baseline_pred = llm.generate_sql(
                question=question,
                db_id=db_id,
                schema_hint=schema_hint,
                reasoning_lessons="[DB-SPECIFIC]\n(none)\n\n[GLOBAL-RULES]\n(none)",
            )
            baseline_meta = _record_llm_call(
                step_idx=i,
                stage="ab_baseline_generate",
                expected_operation="generate_sql",
            )
            baseline_exact, baseline_sim = ast_scores(baseline_pred, gold)
            baseline_pass = baseline_exact or (baseline_sim >= cfg.pass_sim_threshold)
            with_memory_pass = pass1 or pass2

            stats["ab_samples"] += 1
            stats["ab_pass_with_memory"] += 1 if with_memory_pass else 0
            stats["ab_pass_without_memory"] += 1 if baseline_pass else 0
            if with_memory_pass and not baseline_pass:
                stats["ab_with_memory_better"] += 1
            elif baseline_pass and not with_memory_pass:
                stats["ab_without_memory_better"] += 1
            else:
                stats["ab_ties"] += 1

            rec["ab_baseline"] = {
                "pred": baseline_pred,
                "exact": baseline_exact,
                "sim": baseline_sim,
                "pass": baseline_pass,
                "obs": baseline_meta,
            }

        cstats = pipeline_memory_ops.consolidate_memory(
            llm=llm,
            reason_memory=reason_memory,
            global_memory=global_memory,
            step_idx=i + 1,
            every_n_steps=cfg.consolidation_every_n,
            consolidation_k=cfg.consolidation_k,
            verbose=verbose,
        )
        stats["consolidation_merges"] += cstats["merged_reason"] + cstats["merged_global"]
        if kb_store and run_id is not None:
            kb_store.write_step_event(run_id=run_id, rec=rec, stored_bucket=stored_bucket)
            kb_store.persist_snapshot(
                run_id=run_id,
                reason_memory=reason_memory,
                global_memory=global_memory,
                schema_memory=schema_memory,
            )
        _log_observability(
            step_idx=i,
            event_type="outcome",
            payload={
                "db_id": db_id,
                "pass1": bool(pass1),
                "pass2": bool(rec.get("pass2", False)),
                "exact1": bool(exact1),
                "exact2": bool(rec.get("exact2", False)),
                "sim1": float(sim1),
                "sim2": float(rec.get("sim2", 0.0)) if rec.get("sim2") is not None else None,
                "pass_final": bool(pass1 or rec.get("pass2", False)),
                "execution_outcome": "ast_eval_only",
            },
        )

        if verbose and (i + 1) % 10 == 0:
            pass_rate = stats["pass_first"] / max(1, stats["n"])
            print(
                f"--- After {i+1} q: pass_first={pass_rate:.2%}, "
                f"reason_lessons={reason_memory.size}, global_rules={global_memory.size} ---"
            )

        history.append(rec)

    stats["overall_pass_rate"] = stats["pass_final"] / max(1, stats["n"])
    stats["first_try_accuracy"] = stats["pass_first"] / max(1, stats["n"])
    stats["retry_recovery_rate"] = stats["pass_retry"] / max(1, stats["fail_first"])
    stats["exact_match_rate_first"] = stats["exact_first"] / max(1, stats["n"])
    stats["exact_match_rate_final"] = stats["exact_final"] / max(1, stats["n"])
    stats["avg_sim1"] = stats["sim1_sum"] / max(1, stats["n"])
    stats["avg_sim2"] = stats["sim2_sum"] / max(1, stats["sim2_count"])
    stats["sim_lift"] = stats["sim_lift_sum"] / max(1, stats["sim_lift_count"])
    stats["high_sim_non_exact_rate_final"] = stats["high_sim_non_exact_final"] / max(1, stats["n"])
    stats["ab_pass_rate_with_memory"] = stats["ab_pass_with_memory"] / max(1, stats["ab_samples"])
    stats["ab_pass_rate_without_memory"] = stats["ab_pass_without_memory"] / max(1, stats["ab_samples"])
    stats["avg_llm_latency_ms"] = float(stats.get("llm_latency_ms_sum", 0.0)) / max(1, int(stats.get("llm_calls", 0)))
    stats["db_level_accuracy"] = {
        db_id: {
            "n": int(db_totals[db_id]),
            "final_pass": int(db_final_pass[db_id]),
            "final_accuracy": (db_final_pass[db_id] / max(1, db_totals[db_id])),
            "exact_pass": int(db_exact_pass[db_id]),
            "exact_rate": (db_exact_pass[db_id] / max(1, db_totals[db_id])),
        }
        for db_id in sorted(db_totals.keys())
    }
    phoenix.shutdown()

    return stats, history
