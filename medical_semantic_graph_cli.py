"""Build or inspect the derived multi-relation graph without cloud/Neo4j writes."""
import argparse
import json
from pathlib import Path
from medical_semantic_graph import build_graph
from medical_semantic_retrieval import search_semantic_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "query"))
    parser.add_argument("question", nargs="?")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--hops", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        result = build_graph(args.root, args.index)
    else:
        if not args.question:
            parser.error("query requires a question")
        evidence, report = search_semantic_graph(args.question, project_root=args.root,
                                                 path=args.index, max_hops=args.hops)
        result = {"report": report, "excerpts": evidence}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
