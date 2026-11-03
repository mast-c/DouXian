# -*- coding: utf-8 -*-
"""P3-06.3 AST-safe integration into existing trace['knowledge_evidence'].

Does not touch Streamlit page source: existing card display can see appended
display-only graph path cards via the current trace JSON.

Dry-run by default; --apply backs up original rag.py byte-for-byte and does
an atomic replace. Reject unrecognized code and already-patched files.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

MODULE="medical_p3063_inline_path_evidence_v117"
FUNC="_p3063_attach_multihop_path_cards"
IMPORT=f"from {MODULE} import attach_multihop_path_cards as {FUNC}\n"

class InlinePatchBlocked(RuntimeError):
    pass


def _is_trace_evidence(target:ast.AST) -> bool:
    if not isinstance(target,ast.Subscript):
        return False
    return (
        isinstance(target.value,ast.Name)
        and target.value.id=="trace"
        and isinstance(target.slice,ast.Constant)
        and target.slice.value=="knowledge_evidence"
    )


def transform(text:str) -> str:
    """Patch only the known V1.17 RagService._stream_agent assignments."""
    if MODULE in text or FUNC in text:
        raise InlinePatchBlocked("P3-06.3 已安装，拒绝重复插入")
    tree=ast.parse(text)
    imports=[
        n for n in tree.body
        if isinstance(n,ast.ImportFrom)
        and n.module in (
            "medical_p3045_graph_disease_retrieval_v117",
            "medical_p3061_multihop_graph_v117",
        )
    ]
    if not any(
        n.module=="medical_p3045_graph_disease_retrieval_v117"
        for n in imports
    ):
        raise InlinePatchBlocked(
            "未发现 P3-04.5 图谱来源检索接口，"
            "需先安装 P3-04.5，不能直接覆盖 rag.py"
        )
    classes=[
        n for n in tree.body
        if isinstance(n,ast.ClassDef) and n.name=="RagService"
    ]
    if len(classes)!=1:
        raise InlinePatchBlocked("预期唯一的 RagService 类未找到")
    methods=[
        n for n in classes[0].body
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
        and n.name=="_stream_agent"
    ]
    if len(methods)!=1:
        raise InlinePatchBlocked("未发现唯一 RagService._stream_agent")
    method=methods[0]
    assignments=[
        n for n in ast.walk(method)
        if isinstance(n,(ast.Assign,ast.AnnAssign))
        and any(
            _is_trace_evidence(target)
            for target in (
                n.targets if isinstance(n,ast.Assign) else [n.target]
            )
        )
    ]
    # V1.17: 1st builds initial trace from merged_excerpts, 2nd may overwrite
    # from Self-RAG guard. Both must reattach the cached display card list.
    if len(assignments)!=2:
        raise InlinePatchBlocked(
            f"找到 {len(assignments)} 处知识库证据写入，"
            "预期恰好2处，拒绝对未知版本猜测注入位置"
        )
    lines=text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        raise InlinePatchBlocked("rag.py 缺少文件末尾换行")
    additions=[]
    for assign in assignments:
        begin_line=lines[assign.lineno-1]
        indentation=begin_line[:len(begin_line)-len(begin_line.lstrip(" \t"))]
        if not indentation or "\t" in indentation:
            raise InlinePatchBlocked("证据写入处没有可验证的空格缩进")
        hook=(
            f"{indentation}{FUNC}(\n"
            f"{indentation}    trace, question,\n"
            f"{indentation}    project_root=Path(__file__).resolve().parent,\n"
            f"{indentation})\n"
        )
        additions.append((assign.end_lineno,hook))
    anchor=next(
        n for n in imports
        if n.module=="medical_p3045_graph_disease_retrieval_v117"
    )
    additions.append((anchor.end_lineno,IMPORT))
    # Insert in reverse line position, preserving all original lines.
    for index,piece in sorted(additions,key=lambda value:value[0],reverse=True):
        lines.insert(index,piece)
    result="".join(lines)
    ast.parse(result)
    if result.count(f"{FUNC}(")!=2:
        raise InlinePatchBlocked("P3-06.3 证据追加钩子数量异常")
    return result


def patch(target:Path,*,apply:bool=False) -> dict:
    target=Path(target).resolve()
    if not target.is_file():
        raise InlinePatchBlocked(f"rag.py 不存在：{target}")
    for required in (
        "medical_p3063_inline_path_evidence_v117.py",
        "medical_p3061_multihop_graph_v117.py",
        "medical_p3045_graph_disease_retrieval_v117.py",
    ):
        p=target.with_name(required)
        if not p.is_file():
            raise InlinePatchBlocked(f"缺少集成依赖模块：{required}")
        ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    original_raw=target.read_bytes()
    original_hash=hashlib.sha256(original_raw).hexdigest()
    with_bom=original_raw.startswith(b"\xef\xbb\xbf")
    source=original_raw.decode("utf-8-sig")
    lineending="\r\n" if "\r\n" in source else "\n"
    output=transform(source.replace("\r\n","\n"))
    output_raw=output.replace("\n",lineending).encode(
        "utf-8-sig" if with_bom else "utf-8"
    )
    summary={
        "status":"APPLIED" if apply else "READY_TO_APPLY",
        "target":str(target),
        "previous_sha256":original_hash,
        "new_sha256":hashlib.sha256(output_raw).hexdigest(),
        "backup":None,
        "new_source_evidence_rows":0,
        "path_cards_added_to_trace_only":True,
        "medical_generation_unchanged":True,
        "graph_writes":0,
        "chroma_writes":0,
    }
    if not apply:
        return summary
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup=target.with_name(f"rag.py.pre_p3063_{stamp}.bak")
    shutil.copy2(target,backup)
    if hashlib.sha256(backup.read_bytes()).hexdigest()!=original_hash:
        raise InlinePatchBlocked("rag.py 备份哈希校验失败")
    handle,tempfile_name=tempfile.mkstemp(
        prefix="rag_p3063_",suffix=".tmp",dir=str(target.parent)
    )
    try:
        with os.fdopen(handle,"wb") as f:
            f.write(output_raw)
            f.flush()
            os.fsync(f.fileno())
        ast.parse(Path(tempfile_name).read_text(encoding="utf-8-sig"))
        if hashlib.sha256(target.read_bytes()).hexdigest()!=original_hash:
            raise InlinePatchBlocked("rag.py 在补丁生成期间被其他程序修改")
        os.replace(tempfile_name,target)
    finally:
        if os.path.exists(tempfile_name):
            os.unlink(tempfile_name)
    summary["backup"]=str(backup)
    return summary


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",type=Path,default=Path("rag.py"))
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args(argv)
    try:
        data=patch(args.target,apply=args.apply)
        print("[P3-06.3]",data["status"])
        for k,v in data.items():
            if k!="status":print(k+":",v)
        return 0
    except (InlinePatchBlocked,OSError,SyntaxError,ValueError,UnicodeError) as e:
        print("[P3-06.3] BLOCKED:",str(e),file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
