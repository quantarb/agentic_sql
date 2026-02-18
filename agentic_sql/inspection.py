import json
from collections import Counter
from typing import Any, Dict, List

from agentic_sql.memory import GlobalBusinessRulesMemory, ReasoningMemory, SchemaMemory


def memory_summary(
    schema_memory: SchemaMemory,
    reason_memory: ReasoningMemory,
    global_memory: GlobalBusinessRulesMemory | None = None,
) -> Dict[str, int]:
    out = {
        "reason_lessons": len(reason_memory.lessons),
        "reason_index_size": reason_memory.size,
        "dbs_with_schema": len(schema_memory.db_ids_with_memory()),
    }
    if global_memory is not None:
        out["global_rules"] = len(global_memory.lessons)
        out["global_index_size"] = global_memory.size
    return out


def reasoning_error_distribution(lessons: List[str]) -> Dict[str, int]:
    labels: List[str] = []
    for text in lessons:
        try:
            obj = json.loads(text)
            labels.append(obj.get("error_type") or obj.get("title") or "unknown")
        except Exception:
            labels.append("invalid_json")

    return dict(Counter(labels))


def find_duplicate_lessons(lessons: List[str]) -> List[tuple[int, int]]:
    seen: Dict[str, int] = {}
    duplicates: List[tuple[int, int]] = []

    for i, text in enumerate(lessons):
        if text in seen:
            duplicates.append((seen[text], i))
        else:
            seen[text] = i

    return duplicates


def pretty_print_lessons(reason_memory: ReasoningMemory, limit: int = 20) -> None:
    print("Total reasoning lessons:", len(reason_memory.lessons))
    for i, (text, meta) in enumerate(zip(reason_memory.lessons, reason_memory.meta), 1):
        if i > limit:
            print("... (truncated)")
            break

        print("=" * 80)
        print(f"Reasoning Lesson #{i} meta={meta}")
        try:
            print(json.dumps(json.loads(text), indent=2))
        except Exception:
            print(text)
