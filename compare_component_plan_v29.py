# -*- coding: utf-8 -*-
"""V2.9 A/B 日志比较（纯文本离线分析；每个日志最好只有一轮知识库问答）。

示例：
  python compare_component_plan_v29.py --off v29_off.log --on v29_on.log
不会接入模型、不会触碰原始知识库。
"""
import argparse
import json
from pathlib import Path
import re


RE_DURATION=r"(\d+(?:\.\d+)?)"
ANSI=re.compile(r"\x1b\[[0-9;]*m")
LOG_KEY={
    "query_plan_s":r"\[RAG耗时\] 查询分解（规则/Qwen）：" + RE_DURATION + r"s",
    "kb_retrieval_s":r"\[RAG耗时\] 知识库检索总计（到证据确定为止）：" + RE_DURATION + r"s",
    "generation_s":r"\[RAG耗时\] 最终答案流式生成总计：" + RE_DURATION + r"s",
    "model_first_token_s":r"\[RAG耗时\] 最终回答首字等待：" + RE_DURATION + r"s",
}


def last_float(text,expression):
    matches=re.findall(expression,text)
    return float(matches[-1]) if matches else None


def from_log(content):
    content=ANSI.sub("",content)
    result={key:last_float(content,expression) for key,expression in LOG_KEY.items()}
    plan_calls=len(re.findall(LOG_KEY["query_plan_s"],content))
    result["query_count"]=plan_calls
    result["rerank_tokens"]=sum(
        int(x) for x in re.findall(
            r"\[Rerank成本\][^\r\n]*?官方total_tokens=(\d+)",content
        )
    )
    result["rerank_calls"]=len(re.findall(r"\[Rerank成本\]",content))
    result["fast_component_hit"]="[Query Planner] component_fast_path=on, skipped Qwen" in content
    result["graph_statuses"]=re.findall(
        r"\[Graph Assist\] status=(\w+)",content
    )
    s=result.get("kb_retrieval_s")
    g=result.get("generation_s")
    result["approx_rag_total_s"]=round(s+g,2) if s is not None and g is not None else None
    evidence=[]
    section=content.split("========== 最终证据 ==========")
    if len(section)>1:
        body=section[-1].split("[Evidence Guard]",1)[0]
        evidence=re.findall(r"^\[\d+\]\s+(.+)$",body,re.MULTILINE)
    result["selected_sections"]=evidence
    subqueries=re.findall(r"^子问题\s+\d+:\s*(.*)$",content,re.MULTILINE)
    result["subqueries"]=subqueries
    result["warn"]=[]
    if plan_calls!=1:
        result["warn"].append(f"检测到 {plan_calls} 次查询分解，请每份日志只运行一道题。")
    if result["kb_retrieval_s"] is None or result["generation_s"] is None:
        result["warn"].append("缺失检索总计或生成耗时，无法估算整体链路耗时。")
    return result


def _format_number(n,suffix=""):
    return ("未找到" if n is None else f"{n:.2f}{suffix}" if isinstance(n,float)
            else str(n)+suffix)


def display_compare(off,on):
    title=f"{'指标':<31} {'规则 OFF':>14} {'规则 ON':>14}"
    print(title)
    print("-" * len(title))
    entries=[
        ("查询分解耗时","query_plan_s","s"),
        ("检索总计","kb_retrieval_s","s"),
        ("最终流式生成","generation_s","s"),
        ("估算问答链路耗时","approx_rag_total_s","s"),
        ("模型首字生成等待","model_first_token_s","s"),
        ("Rerank 调用次数","rerank_calls",""),
        ("Rerank 官方 Tokens","rerank_tokens",""),
        ("命中 V2.9 Python 规则","fast_component_hit",""),
    ]
    for label,key,units in entries:
        print(f"{label:<31} {_format_number(off.get(key),units):>14} {_format_number(on.get(key),units):>14}")
    q1,q2=off.get("query_plan_s"),on.get("query_plan_s")
    if q1 is not None and q2 is not None:
        print(f"\n本轮查询规划差：{q1-q2:+.2f}s（正数表示开启后节省时间）。")
    for label,data in (("OFF",off),("ON",on)):
        print(f"\n{label} 的最终证据章节：")
        for section in data["selected_sections"]:
            print(" -",section)
        for warning in data["warn"]:
            print(" ⚠",warning)
    print("\n注意：不同子查询可能改变证据与 Token；上述日志不能独立证明引用的事实准确性。")
    print("计时只是近似问答链路；不包括完整 Streamlit 网页体感/启动开销。")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off",type=Path,required=True)
    ap.add_argument("--on",type=Path,required=True)
    ap.add_argument("--save-json",type=Path,default=None,help="可选：输出汇总 JSON")
    args=ap.parse_args(argv)
    for label,path in (("OFF",args.off),("ON",args.on)):
        if not path.is_file():
            ap.error(f"{label} 日志不存在：{path}")
    off=from_log(args.off.read_text(encoding="utf-8-sig",errors="replace"))
    on=from_log(args.on.read_text(encoding="utf-8-sig",errors="replace"))
    display_compare(off,on)
    if args.save_json:
        args.save_json.write_text(
            json.dumps({"off":off,"on":on},ensure_ascii=False,indent=2),
            encoding="utf-8")
        print("\n汇总文件：",args.save_json)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
