# -*- coding: utf-8 -*-
"""Read-only diagnosis of actual V2.11 JSONL chunk snapshot and graph paths.

Use a UTF-8 .py file, not a PowerShell here-string piped to `python -`.
All terminal output is ASCII-safe, even on legacy Windows GBK terminals.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path

if __package__:
    from . import graph_chunk_linker_v211 as linker
    from .graph_paths import BINDINGS_DIR, CONTENT_DIR
else:
    import graph_chunk_linker_v211 as linker
    from graph_paths import BINDINGS_DIR, CONTENT_DIR

canonical_section = linker.canonical_section
canonical_source = linker.canonical_source
norm = linker.norm
quote_for_matching = linker.quote_for_matching
read_snapshot = linker.read_snapshot
SEMANTIC = linker.SEMANTIC


def ascii_json(value):
    return json.dumps(value,ensure_ascii=True,separators=(",",":"))


def inspect(graph_file,chunk_file):
    graph=json.loads(Path(graph_file).read_text(encoding="utf-8-sig"))
    records=read_snapshot(Path(chunk_file))
    groups=OrderedDict()
    for edge in graph["edges"]:
        if edge.get("relation") not in SEMANTIC:
            continue
        ev=edge["evidence"]
        quote=quote_for_matching(edge)
        groups.setdefault((ev["source_file"],ev["section_path"]),quote)

    inspected=0
    exact=0
    still_conflict=0
    no_quote=0
    for (filename,path),quote in groups.items():
        inspected+=1
        target=canonical_section(path,filename)
        matches=[
            row for row in records
            if canonical_source(row["source"])==canonical_source(filename)
            and norm(quote) in norm(row["content"])
        ]
        print("\nGRAPH_SECTION="+ascii_json(path))
        print("QUOTE_FOUND_CHUNKS="+str(len(matches)))
        if not matches:
            no_quote+=1
            print("STATUS=NO_EXACT_QUOTE")
        for row in matches[:3]:
            actual=row["section_path"]
            ok=canonical_section(actual,filename)==target
            raw_equal=canonical_section(actual)==canonical_section(path)
            if ok:
                exact+=1
            else:
                still_conflict+=1
            print("CHUNK_ID="+ascii_json(row["chunk_id"]))
            print("CHROMA_SECTION="+ascii_json(actual))
            print("OLD_SECTION_EQUAL="+str(raw_equal))
            print("PATCHED_SECTION_EQUAL="+str(ok))
            print("STATUS="+("EXACT" if ok else "STILL_SECTION_CONFLICT"))
        if len(matches)>3:
            print("MORE_MATCHED_CHUNKS="+str(len(matches)-3))

    print("\nSUMMARY_SECTION_GROUPS="+str(inspected))
    print("MATCHED_EXAMPLES="+str(exact))
    print("UNRESOLVED_EXAMPLES="+str(still_conflict))
    print("GROUPS_WITHOUT_QUOTE="+str(no_quote))
    print("SOURCE_SNAPSHOT_ROWS="+str(len(records)))
    print("READ_ONLY=TRUE")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",default=str(CONTENT_DIR / "knowledge_graph_v27.json"))
    parser.add_argument("--chunks",default=str(BINDINGS_DIR / "chroma_chunks_v211.json"))
    args=parser.parse_args(argv)
    inspect(args.graph,args.chunks)


if __name__=="__main__":
    main()
