"""只读 Neo4j + 真实云端药品排序验收；不执行任何知识库写入。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="治疗高血压最常用的5种药物")
    parser.add_argument("--allow-cloud-query", action="store_true")
    parser.add_argument("--live-guideline-search", action="store_true", help="缓存缺失时允许额外联网，默认仅复用近期摘要")
    parser.add_argument("--output", type=Path, default=ROOT/"medical_drug_ranking_smoke.json")
    args = parser.parse_args()
    if not args.allow_cloud_query:
        parser.error("真实模型验收需要 --allow-cloud-query")
    if args.output.exists():
        parser.error("输出文件已存在，请使用新的 --output 路径")
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from drug_graph_retriever import DrugGraphRetriever
    from medical_drug_ranking import rank_drug_candidates, requested_drug_count
    from search_tool import search_guideline_summary
    load_dotenv(ROOT/".env", override=False)
    started = time.monotonic()
    retriever = DrugGraphRetriever()
    try:
        rows, pool = retriever.search_disease_drug_pool(args.question)
    finally:
        retriever.close()
    model_name = os.getenv("OPENAI_MODEL")
    options = {"extra_body": {"enable_thinking": False}} if "qwen" in str(model_name).lower() else {}
    model = ChatOpenAI(model=model_name, api_key=os.getenv("DASHSCOPE_API_KEY"),
                       base_url=os.getenv("OPENAI_BASE_URL"), timeout=40, max_retries=0,
                       max_tokens=min(12000, requested_drug_count(args.question) * 300 + 1000), **options)

    responses = []
    def ask(system, payload):
        content = model.invoke([SystemMessage(content=system), HumanMessage(
            content=json.dumps(payload, ensure_ascii=False))]).content
        responses.append(content)
        return content

    selected, report = rank_drug_candidates(args.question, rows, ask=ask,
        search=search_guideline_summary if args.live_guideline_search else None)
    report.update(pool=pool, question=args.question, elapsed_seconds=round(time.monotonic()-started, 2),
                  model_responses=responses,
                  smoke_test_only=True, selected=[{
                      "drug_name": r["drug_name"], "candidate_id": r["candidate_id"],
                      "ranking_basis": r["ranking_basis"], "support_quote": r["ranking_support_quote"],
                      "source_url": r["source_url"],
                  } for r in selected])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("guideline_summary", "model_responses")}, ensure_ascii=True))
    return 0 if report["status"] == "ranked" and not report["shortfall"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
