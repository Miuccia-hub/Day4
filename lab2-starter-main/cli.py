"""Command-line entry point for indexing, search, and report export."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_core import EvidenceStore, build_due_diligence_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Canvassian M&A evidence review")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="build or refresh the evidence index").add_argument("--force", action="store_true")
    search = sub.add_parser("search", help="search the evidence corpus")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--category", choices=["board_papers", "contracts", "emails"])
    report = sub.add_parser("report", help="generate the Board report")
    report.add_argument("--output", default="outputs/canvassian_due_diligence_report.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    store = EvidenceStore(root)
    stats = store.ensure_index(force=getattr(args, "force", False))
    if args.command == "index":
        print(stats)
    elif args.command == "search":
        for result in store.search(args.query, args.top_k, args.category):
            print(f"\n[{result.score:.3f}] {result.citation}\n{result.chunk.text[:700]}")
    else:
        destination = root / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(build_due_diligence_report(store), encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()

