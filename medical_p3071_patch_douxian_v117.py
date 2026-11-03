# -*- coding: utf-8 -*-
"""P3-07.1 AST-safe, reversible evidence-panel integration into 豆馅.py.

Dry run by default. --apply backs up the byte-exact main and atomically adds
read-only interaction source-chain rendering inside existing HTML evidence.
NEVER writes rag.py, Neo4j, Chroma, or medical.json.
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

IMPORT="from medical_p3071_interaction_ui_v117 import (prepare_interaction_trace_for_display, interaction_evidence_html)\n"
AFTER_PATH_DISPLAY="""    multihop_html = multihop_path_evidence_html(trace)
    if multihop_html:
        content.append(multihop_html)
"""
REPLACE_HTML="""    multihop_html = multihop_path_evidence_html(trace)
    if multihop_html:
        content.append(multihop_html)

    # P3-07.1：华法林×布洛芬的来源链，紧接原文证据和多跳路径。
    # 与知识库命中/生成引用的医学证据统计彻底隔离。
    p3071_html = interaction_evidence_html(trace)
    if p3071_html:
        content.append(p3071_html)
"""
AFTER_POPULATE="""    populate_multihop_trace_for_display(trace, pending)
"""
REPLACE_POPULATE="""    populate_multihop_trace_for_display(trace, pending)
    # P3-07.1：仅在最终回答生成后缓存可追溯双药来源链。
    prepare_interaction_trace_for_display(
        trace, pending, project_root=Path(__file__).resolve().parent
    )
"""


class PatchBlocked(RuntimeError):
    pass


def transform(source:str)->str:
    if "medical_p3071_interaction_ui_v117" in source:
        raise PatchBlocked("P3-07.1 已存在，拒绝重复打补丁")
    tree=ast.parse(source)
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    needed={"trace_details_html","populate_multihop_trace_for_display"}
    if not needed.issubset(names):
        raise PatchBlocked("此版本没有已验证的 P3-06.3 HTML 证据面板结构")
    if "trace_details_html(trace)" not in source:
        raise PatchBlocked("找不到聊天证据 HTML 展示入口")
    if source.count(AFTER_PATH_DISPLAY)!=1 or source.count(AFTER_POPULATE)!=1:
        raise PatchBlocked(
            "前端多跳路径/会话缓存位置不是唯一且已知的形态，"
            "拒绝盲目修改你的 Streamlit 主页面"
        )
    imports=[
        n for n in tree.body
        if isinstance(n,ast.Import)
        and any(alias.name=="re" and alias.asname is None for alias in n.names)
    ]
    if len(imports)!=1:
        raise PatchBlocked("缺少唯一顶层 import re；先应用 P3-06.3C NameError 热修复")
    old=source
    source=source.replace(AFTER_PATH_DISPLAY,REPLACE_HTML,1)
    source=source.replace(AFTER_POPULATE,REPLACE_POPULATE,1)
    eol="\r\n" if "\r\n" in old else "\n"
    # AST line splice must happen after content replacements.
    lines=source.splitlines(keepends=True)
    idx=imports[0].end_lineno
    lines.insert(idx,IMPORT.replace("\n",eol))
    result="".join(lines)
    ast.parse(result)
    if result.count("interaction_evidence_html(trace)")!=1:
        raise PatchBlocked("新的前端证据链渲染钩子数量异常")
    if result.count("prepare_interaction_trace_for_display(")!=1:
        raise PatchBlocked("新的只读检索钩子数量异常")
    return result


def patch(file:Path,apply:bool=False)->dict:
    file=Path(file).resolve()
    if not file.is_file():
        raise PatchBlocked("豆馅.py 不存在")
    module=file.with_name("medical_p3071_interaction_ui_v117.py")
    core=file.with_name("medical_p3071_drug_interaction_v117.py")
    for dep in (module,core):
        if not dep.is_file():
            raise PatchBlocked(f"运行依赖缺失：{dep.name}")
        ast.parse(dep.read_text("utf-8-sig"))
    raw=file.read_bytes()
    bom=raw.startswith(b"\xef\xbb\xbf")
    source=raw.decode("utf-8-sig")
    lineending="\r\n" if "\r\n" in source else "\n"
    normalized=source.replace("\r\n","\n")
    changed=transform(normalized).replace("\n",lineending)
    data=changed.encode("utf-8-sig" if bom else "utf-8")
    before=hashlib.sha256(raw).hexdigest()
    info={
        "status":"READY_TO_APPLY" if not apply else "APPLIED",
        "target":str(file),
        "before_sha256":before,
        "after_sha256":hashlib.sha256(data).hexdigest(),
        "backup":None,
        "database_writes":0,
        "model_response_code_changed":False,
        "medically_verified":False,
    }
    if not apply:return info
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup=file.with_name(f"豆馅.py.pre_p3071_{timestamp}.bak")
    shutil.copy2(file,backup)
    if backup.read_bytes()!=raw:
        raise PatchBlocked("备份字节校验失败，拒绝覆盖")
    fd,tmp=tempfile.mkstemp(
        prefix="douxian_p3071_",suffix=".tmp",dir=str(file.parent)
    )
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        ast.parse(Path(tmp).read_text("utf-8-sig"))
        if file.read_bytes()!=raw:
            raise PatchBlocked("豆馅.py 安装过程中被修改，拒绝覆盖")
        os.replace(tmp,file)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    info["backup"]=str(backup)
    return info


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",type=Path,default=Path("豆馅.py"))
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args(argv)
    try:
        info=patch(args.target,apply=args.apply)
        print("[P3-07.1]",info["status"])
        for key,val in info.items():
            if key!="status":print(key+":",val)
        return 0
    except (OSError,ValueError,RuntimeError,SyntaxError,UnicodeError) as ex:
        print("[P3-07.1] BLOCKED:",ex,file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
