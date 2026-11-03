# -*- coding: utf-8 -*-
"""P3-07.2 safe AST-compatible patch for the EXISTING original Douxian evidence UI.

Does not create a standalone Streamlit page. Dry-run by default, apply
creates byte-exact backup and atomically replaces frontend only.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

MODULE="medical_p3072_interaction_ui_v117"
IMPORT="""from medical_p3072_interaction_ui_v117 import (
    prepare_full_corpus_interaction_trace, interaction_full_corpus_html
)
"""
PATH_ANCHOR="""    multihop_html = multihop_path_evidence_html(trace)
    if multihop_html:
        content.append(multihop_html)
"""
PATH_REPLACEMENT=PATH_ANCHOR+"""
    # P3-07.2：全库双药安全来源链只显示在同一个原文证据HTML区域。
    # 未验证的来源关联绝不增加生成引用计数或 Neo4j 关系数量。
    p3072_html = interaction_full_corpus_html(trace)
    if p3072_html:
        content.append(p3072_html)
"""
POST_ANSWER="""    populate_multihop_trace_for_display(trace, pending)
"""
POST_REPLACEMENT=POST_ANSWER+"""    prepare_full_corpus_interaction_trace(
        trace, pending, root=Path(__file__).resolve().parent
    )
"""


class PatchBlocked(RuntimeError):
    pass


def transform(source:str)->str:
    if MODULE in source:
        raise PatchBlocked("P3-07.2 已接入，拒绝重复打补丁")
    tree=ast.parse(source)
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    if not {"trace_details_html","populate_multihop_trace_for_display"}.issubset(names):
        raise PatchBlocked("当前豆馅.py 缺少 P3-06.3 HTML 原证据栏结构")
    if source.count(PATH_ANCHOR)!=1 or source.count(POST_ANSWER)!=1:
        raise PatchBlocked("原证据区域或生成后 trace 钩子不是唯一，拒绝覆盖未知版本")
    imports=[
        n for n in tree.body
        if isinstance(n,ast.Import)
        and any(a.name=="re" and a.asname is None for a in n.names)
    ]
    if len(imports)!=1:
        raise PatchBlocked("缺少唯一顶层 import re；请先安装 P3-06.3C 修复")
    source=source.replace(PATH_ANCHOR,PATH_REPLACEMENT,1)
    source=source.replace(POST_ANSWER,POST_REPLACEMENT,1)
    lines=source.splitlines(keepends=True)
    lines.insert(imports[0].end_lineno,IMPORT)
    patched="".join(lines)
    ast.parse(patched)
    assert patched.count("interaction_full_corpus_html(trace)")==1
    assert patched.count("prepare_full_corpus_interaction_trace(")==1
    return patched


def patch(target:Path,*,apply:bool=False)->dict:
    target=Path(target).resolve()
    if not target.is_file():
        raise PatchBlocked(f"缺少豆馅.py：{target}")
    for name in ("medical_p3072_interaction_ui_v117.py",
                 "medical_p3072_corpus_index_v117.py",
                 "medical_p3072_pair_retrieval_v117.py"):
        dep=target.with_name(name)
        if not dep.is_file():
            raise PatchBlocked(f"缺少 P3-07.2 依赖：{name}")
        ast.parse(dep.read_text("utf-8-sig"),filename=name)
    raw=target.read_bytes()
    bom=raw.startswith(b"\xef\xbb\xbf")
    old=raw.decode("utf-8-sig")
    normalized=old.replace("\r\n","\n")
    eol="\r\n" if "\r\n" in old else "\n"
    changed=transform(normalized).replace("\n",eol)
    data=changed.encode("utf-8-sig" if bom else "utf-8")
    info={
        "status":"APPLIED" if apply else "READY_TO_APPLY",
        "source":str(target),"original_sha256":hashlib.sha256(raw).hexdigest(),
        "new_sha256":hashlib.sha256(data).hexdigest(),"backup":None,
        "rag_py_changed":False,"neo4j_writes":0,"chroma_writes":0,
    }
    if not apply:return info
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup=target.with_name(f"豆馅.py.pre_p3072_{stamp}.bak")
    shutil.copy2(target,backup)
    if backup.read_bytes()!=raw:
        raise PatchBlocked("备份字节核验未通过")
    fd,tmp=tempfile.mkstemp(
        dir=target.parent,prefix="douxian_p3072_",suffix=".tmp"
    )
    try:
        with os.fdopen(fd,"wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        ast.parse(Path(tmp).read_text("utf-8-sig"))
        if target.read_bytes()!=raw:
            raise PatchBlocked("安装期间其他程序修改了豆馅.py，拒绝覆盖")
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    info["backup"]=str(backup)
    return info


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target",type=Path,default=Path("豆馅.py"))
    ap.add_argument("--apply",action="store_true")
    args=ap.parse_args(argv)
    try:
        result=patch(args.target,apply=args.apply)
        print("[P3-07.2]",result["status"])
        for key,val in result.items():
            if key!="status":print(f"{key}: {val}")
        return 0
    except (ValueError,RuntimeError,OSError,SyntaxError,UnicodeError) as exc:
        print("[P3-07.2] BLOCKED:",exc,file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
