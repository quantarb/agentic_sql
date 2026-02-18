import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from agentic_sql import (
    ExperimentConfig,
    ModelConfig,
    build_runtime,
    require_openai_api_key,
    run_dual_memory_experiment,
)


@dataclass(frozen=True)
class EvalThresholds:
    min_overall_pass_delta: float = 0.0
    min_exact_final_delta: float = 0.0
    max_cost_increase_ratio: float = 0.25
    max_latency_increase_ratio: float = 0.30


def _metric(stats: Dict[str, Any], key: str) -> float:
    try:
        return float(stats.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _run_variant(
    *,
    name: str,
    examples: List[Dict[str, Any]],
    exp_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
    db_path: str,
    verbose: bool,
) -> Dict[str, Any]:
    runtime = build_runtime(
        db_path=db_path,
        model_cfg=model_cfg,
        bootstrap_from_kb=False,
    )

    stats, _history = run_dual_memory_experiment(
        examples=examples,
        llm=runtime.llm,
        schema_memory=runtime.schema_memory,
        reason_memory=runtime.reason_memory,
        global_memory=runtime.global_memory,
        kb_store=runtime.kb_store,
        cfg=exp_cfg,
        verbose=verbose,
    )
    return {"name": name, "stats": stats}


def _evaluate_checks(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    thresholds: EvalThresholds,
) -> List[Dict[str, Any]]:
    b = baseline["stats"]
    c = candidate["stats"]
    checks: List[Dict[str, Any]] = []

    pass_delta = _metric(c, "overall_pass_rate") - _metric(b, "overall_pass_rate")
    checks.append(
        {
            "name": "overall_pass_rate_delta",
            "value": pass_delta,
            "threshold": thresholds.min_overall_pass_delta,
            "ok": pass_delta >= thresholds.min_overall_pass_delta,
        }
    )

    exact_delta = _metric(c, "exact_match_rate_final") - _metric(b, "exact_match_rate_final")
    checks.append(
        {
            "name": "exact_match_rate_final_delta",
            "value": exact_delta,
            "threshold": thresholds.min_exact_final_delta,
            "ok": exact_delta >= thresholds.min_exact_final_delta,
        }
    )

    base_cost = max(_metric(b, "llm_estimated_cost_usd"), 1e-12)
    cand_cost = _metric(c, "llm_estimated_cost_usd")
    cost_ratio = (cand_cost - base_cost) / base_cost
    checks.append(
        {
            "name": "cost_increase_ratio",
            "value": cost_ratio,
            "threshold": thresholds.max_cost_increase_ratio,
            "ok": cost_ratio <= thresholds.max_cost_increase_ratio,
        }
    )

    base_lat = max(_metric(b, "avg_llm_latency_ms"), 1e-12)
    cand_lat = _metric(c, "avg_llm_latency_ms")
    lat_ratio = (cand_lat - base_lat) / base_lat
    checks.append(
        {
            "name": "latency_increase_ratio",
            "value": lat_ratio,
            "threshold": thresholds.max_latency_increase_ratio,
            "ok": lat_ratio <= thresholds.max_latency_increase_ratio,
        }
    )
    return checks


def _to_markdown(report: Dict[str, Any]) -> str:
    baseline = report["baseline"]["stats"]
    candidate = report["candidate"]["stats"]
    lines = [
        "# Eval Harness Report",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- dataset: {report['dataset']} split={report['split']} n_steps={report['n_steps']}",
        f"- gates_passed: {report['gates_passed']}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_keys = [
        "overall_pass_rate",
        "exact_match_rate_final",
        "retry_recovery_rate",
        "avg_sim1",
        "avg_sim2",
        "sim_lift",
        "avg_llm_latency_ms",
        "llm_estimated_cost_usd",
        "llm_total_tokens",
    ]
    for k in metric_keys:
        b = _metric(baseline, k)
        c = _metric(candidate, k)
        lines.append(f"| {k} | {b:.6f} | {c:.6f} | {c - b:+.6f} |")

    lines.extend(["", "## Gates", ""])
    for chk in report["checks"]:
        status = "PASS" if chk["ok"] else "FAIL"
        lines.append(
            f"- {status} `{chk['name']}` value={chk['value']:.6f} threshold={chk['threshold']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs candidate eval harness for Agentic SQL.")
    parser.add_argument("--dataset", default="xlangai/spider")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--db-path", default="knowledge_base.db")
    parser.add_argument("--output-json", default="eval_report.json")
    parser.add_argument("--output-md", default="eval_report.md")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--min-overall-pass-delta", type=float, default=0.0)
    parser.add_argument("--min-exact-final-delta", type=float, default=0.0)
    parser.add_argument("--max-cost-increase-ratio", type=float, default=0.25)
    parser.add_argument("--max-latency-increase-ratio", type=float, default=0.30)
    args = parser.parse_args()

    require_openai_api_key()

    ds = load_dataset(args.dataset)
    if args.split not in ds:
        raise RuntimeError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    examples = list(ds[args.split])[: args.n_steps]
    model_cfg = ModelConfig()

    # Baseline removes memory/context retrieval to provide a simple control.
    baseline_cfg = ExperimentConfig(
        n_steps=args.n_steps,
        retrieval_k=0,
        initial_retrieval_k=0,
        global_rule_k=0,
        context_top_k=0,
        runtime_context_first_k=0,
        runtime_context_boost=0.0,
        phoenix_enabled=False,
    )
    candidate_cfg = ExperimentConfig(n_steps=args.n_steps)

    baseline = _run_variant(
        name="baseline_no_memory",
        examples=examples,
        exp_cfg=baseline_cfg,
        model_cfg=model_cfg,
        db_path=args.db_path,
        verbose=args.verbose,
    )
    candidate = _run_variant(
        name="candidate_default",
        examples=examples,
        exp_cfg=candidate_cfg,
        model_cfg=model_cfg,
        db_path=args.db_path,
        verbose=args.verbose,
    )

    thresholds = EvalThresholds(
        min_overall_pass_delta=args.min_overall_pass_delta,
        min_exact_final_delta=args.min_exact_final_delta,
        max_cost_increase_ratio=args.max_cost_increase_ratio,
        max_latency_increase_ratio=args.max_latency_increase_ratio,
    )
    checks = _evaluate_checks(baseline=baseline, candidate=candidate, thresholds=thresholds)
    gates_passed = all(bool(c["ok"]) for c in checks)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "n_steps": int(args.n_steps),
        "thresholds": asdict(thresholds),
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "gates_passed": gates_passed,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out_md.write_text(_to_markdown(report))

    print("Wrote:", out_json)
    print("Wrote:", out_md)
    print("Gates passed:", gates_passed)


if __name__ == "__main__":
    main()
