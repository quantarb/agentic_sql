import os

from agentic_sql import build_llm
from agentic_sql.preprocess import OfflineContextBuilder


def main() -> None:
    llm = build_llm()
    builder = OfflineContextBuilder("knowledge_base.db")
    runtime_context_path = os.environ.get("RUNTIME_CONTEXT_PATH")

    stats = builder.build_context(
        run_id=None,
        human_annotations_path="human_annotations.jsonl" if os.path.exists("human_annotations.jsonl") else None,
        docs_dir="institutional_docs",
        code_dir=".",
        runtime_context_path=runtime_context_path,
    )
    embedded = builder.embed_context(embed_fn=llm.embed_texts, emb_model=llm.models.emb_model)

    print("Context build stats:", stats)
    if runtime_context_path:
        print("Runtime context source:", runtime_context_path)
    print("Embedded context items:", embedded)

    sample = builder.retrieve(
        question="How should I join customer tables for active usage?",
        embed_fn=llm.embed_texts,
        top_k=5,
    )
    print("\nSample retrieval:")
    for row in sample:
        print(f"- [{row['layer']}] {row['title']} (score={row['score']:.3f})")


if __name__ == "__main__":
    main()
