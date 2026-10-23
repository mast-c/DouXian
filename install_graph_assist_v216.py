# -*- coding: utf-8 -*-
"""V2.16 Assist Planner additive installer; NEVER patches rag.py.

Strict frozen-project hashes, exclusive file creates, safe receipt-based
uninstall. No Chroma/reindex/LLM/API activity.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parent
PAYLOAD=PROJECT/"v216_payload"
MANIFEST=json.loads((PROJECT/"manifest_v216_assist.json").read_text(encoding="utf-8"))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def check():
    errors=[]
    for rel,expected in MANIFEST["pinned_existing_sha256"].items():
        f=PROJECT/rel
        if not f.is_file():
            errors.append("Missing required frozen original: "+rel)
        elif sha(f)!=expected:
            errors.append("Existing data/code SHA256 drift: "+rel)
    for rel in MANIFEST["required_runtime_files"]:
        if not (PROJECT/rel).is_file():
            errors.append("Missing existing RAG dependency: "+rel)
    for rel,expected in MANIFEST["additive_payload_sha256"].items():
        f=PAYLOAD/rel
        if not f.is_file() or sha(f)!=expected:
            errors.append("Damaged/modified V2.16 payload: "+rel)
        target=PROJECT/rel
        if target.exists() and (not target.is_file() or sha(target)!=expected):
            errors.append("Existing file differs; REFUSE overwrite: "+rel)
    return errors

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true")
    group.add_argument("--install",action="store_true")
    group.add_argument("--uninstall",metavar="INSTALL_RECEIPT")
    args=parser.parse_args()
    if args.uninstall:
        receipt=Path(args.uninstall).expanduser().resolve()
        reportdir=(PROJECT/"graph_reports/assist_v216").resolve()
        if not receipt.is_file() or receipt.parent!=reportdir:
            raise RuntimeError("V2.16 install receipt must be under graph_reports/assist_v216/")
        data=json.loads(receipt.read_text(encoding="utf-8"))
        if Path(data.get("project","")).resolve()!=PROJECT.resolve():
            raise RuntimeError("This install receipt belongs to a different project")
        files=data.get("files_created")
        if not isinstance(files,dict) or not files:
            raise RuntimeError("No installed V2.16 files in receipt")
        for rel,expected in files.items():
            f=PROJECT/rel
            if (rel not in MANIFEST["additive_payload_sha256"]
                or MANIFEST["additive_payload_sha256"][rel]!=expected
                or not f.is_file() or sha(f)!=expected):
                raise RuntimeError("Refuse removal of unrecognized/modified file: "+rel)
        for rel in files:(PROJECT/rel).unlink()
        print("[V2.16] Removed only files installed by this receipt.")
        return
    errors=check()
    if errors:
        raise RuntimeError("\n".join(errors)+"\nV2.16 nothing installed")
    print("[V2.16] preflight PASSED: old source hashes and new files match.")
    if args.check:return
    created={}
    try:
        for rel,expected in MANIFEST["additive_payload_sha256"].items():
            target=PROJECT/rel
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as out:
                # Tracked BEFORE write: partial failure rolls back cleanly.
                created[rel]=expected
                with (PAYLOAD/rel).open("rb") as stream:
                    while True:
                        b=stream.read(1048576)
                        if not b:break
                        out.write(b)
        if created:
            reportdir=PROJECT/"graph_reports/assist_v216"
            reportdir.mkdir(parents=True,exist_ok=True)
            receipt=reportdir/("v216_assist_install_"+
                    datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
            with receipt.open("x",encoding="utf-8") as fd:
                json.dump({"project":str(PROJECT.resolve()),
                           "files_created":created,
                           "phase":"offline_planner_not_online"},
                          fd,ensure_ascii=False,indent=2)
                fd.write("\n")
            print("[V2.16] install receipt:",receipt)
        print("[V2.16] installed planner & experiments only.")
        print("[V2.16] rag.py, graph_assist_v28.py, Graph V2.13 & Chroma unchanged.")
    except Exception:
        for rel in created:
            target=PROJECT/rel
            if target.is_file():target.unlink()
        raise

if __name__=="__main__":
    main()
