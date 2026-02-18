from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    gen_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"
    lesson_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    reranker_model: str = "gpt-4o-mini"
    consolidator_model: str = "gpt-4o-mini"
    schema_model: str = "gpt-4o-mini"
    emb_model: str = "text-embedding-3-small"
    emb_provider: str = "sentence_transformers"  # "sentence_transformers" | "openai"
    local_emb_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ExperimentConfig:
    n_steps: int = 50
    retrieval_k: int = 3
    initial_retrieval_k: int = 20
    global_rule_k: int = 3
    context_top_k: int = 6
    context_per_layer_k: int = 3
    context_max_chars: int = 4000
    qa_top_k: int = 4
    qa_max_chars: int = 1200
    consolidation_k: int = 6
    consolidation_every_n: int = 10
    memory_ab_every_n: int = 0
    pass_sim_threshold: float = 0.92
    debug_trace: bool = False
    debug_context_chars: int = 1200
    min_global_confidence: float = 0.8
    min_global_evidence: int = 2
    tentative_ttl_steps: int = 200
    include_tentative_in_retrieval: bool = False
    runtime_context_first_k: int = 2
    runtime_context_boost: float = 0.15
    runtime_context_max_age_hours: int = 48
    planner_include_context_chars: int = 1200
    phoenix_enabled: bool = False
    phoenix_project_name: str = "agentic-sql"
    phoenix_endpoint: str = ""
