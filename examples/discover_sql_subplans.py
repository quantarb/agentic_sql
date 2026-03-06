import argparse

from agentic_sql.subquery_miner import SQLPatternMiner, iter_sql_from_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover reusable SQL subquery patterns from a query corpus."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SQL corpus. Supports line-delimited SQL text or JSONL.",
    )
    parser.add_argument(
        "--sql-key",
        default="sql",
        help="JSON key for SQL text when --input is JSONL. Default: sql",
    )
    parser.add_argument(
        "--db-path",
        default="sql_pattern_mining.db",
        help="SQLite path used for graph storage and metrics.",
    )
    parser.add_argument("--dialect", default="sqlite", help="SQL dialect for sqlglot parser.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records per SQLite write batch.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="How many top reusable fragments to print.",
    )
    parser.add_argument(
        "--snapshot-json",
        default="sql_patterns_graph.json",
        help="Output JSON graph snapshot path.",
    )
    parser.add_argument(
        "--snapshot-dot",
        default="sql_patterns_graph.dot",
        help="Optional DOT graph path for Graphviz.",
    )
    args = parser.parse_args()

    miner = SQLPatternMiner(db_path=args.db_path, dialect=args.dialect)
    stats = miner.ingest_queries(
        iter_sql_from_file(args.input, sql_key=args.sql_key),
        batch_size=args.batch_size,
    )
    print(f"ingest: processed={stats.processed}, parsed={stats.parsed}, failed={stats.failed}")
    print("graph:", miner.summary())

    pr = miner.run_pagerank()
    print(f"pagerank: iterations={pr['iterations']}, delta={pr['delta']:.6g}")

    print("\nTop reusable fragments:")
    for i, row in enumerate(miner.top_fragments(limit=args.top_k), start=1):
        sql_preview = row["canonical_sql"]
        if len(sql_preview) > 120:
            sql_preview = sql_preview[:117] + "..."
        print(
            f"{i:02d}. kind={row['kind']} support={row['query_support']} "
            f"count={row['raw_count']} pr={row['pagerank']:.6g} | {sql_preview}"
        )

    snap = miner.export_graph_snapshot(
        json_path=args.snapshot_json,
        dot_path=args.snapshot_dot,
    )
    print(
        f"\nGraph snapshot written: nodes={snap['nodes']}, edges={snap['edges']} "
        f"json={args.snapshot_json} dot={args.snapshot_dot}"
    )


if __name__ == "__main__":
    main()
