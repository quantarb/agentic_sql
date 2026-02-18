import os
from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI

from agentic_sql.config import ModelConfig
from agentic_sql.kb_store import KnowledgeBaseStore
from agentic_sql.llm import TextToSQLLLM
from agentic_sql.memory import GlobalBusinessRulesMemory, ReasoningMemory, SchemaMemory


@dataclass(frozen=True)
class AgentRuntime:
    llm: TextToSQLLLM
    schema_memory: SchemaMemory
    reason_memory: ReasoningMemory
    global_memory: GlobalBusinessRulesMemory
    kb_store: KnowledgeBaseStore
    bootstrap_stats: Dict[str, Any]


def require_openai_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    api_key = (os.environ.get(env_var) or "").strip()
    if not api_key:
        raise RuntimeError(f"Set {env_var} before running.")
    return api_key


def build_llm(model_cfg: ModelConfig | None = None) -> TextToSQLLLM:
    require_openai_api_key()
    return TextToSQLLLM(OpenAI(), model_cfg or ModelConfig())


def build_runtime(
    *,
    db_path: str = "knowledge_base.db",
    model_cfg: ModelConfig | None = None,
    bootstrap_from_kb: bool = True,
    run_id: int | None = None,
) -> AgentRuntime:
    llm = build_llm(model_cfg=model_cfg)
    schema_memory = SchemaMemory()
    reason_memory = ReasoningMemory(llm.embed_texts)
    global_memory = GlobalBusinessRulesMemory(llm.embed_texts)
    kb_store = KnowledgeBaseStore(db_path)

    bootstrap_stats: Dict[str, Any] = {}
    if bootstrap_from_kb:
        bootstrap_stats = kb_store.load_into_runtime_memory(
            reason_memory=reason_memory,
            global_memory=global_memory,
            schema_memory=schema_memory,
            run_id=run_id,
        )

    return AgentRuntime(
        llm=llm,
        schema_memory=schema_memory,
        reason_memory=reason_memory,
        global_memory=global_memory,
        kb_store=kb_store,
        bootstrap_stats=bootstrap_stats,
    )
