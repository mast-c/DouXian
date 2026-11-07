# -*- coding: utf-8 -*-
"""Verify the *running deployment* really uses LLM-first medical query parsing.

Usage, in your original Python project root:
  python diagnose_llm_first.py --root . --question "..."
Reads .env in the working directory, script directory or their parents.
Supports the original OPENAI_BASE_URL / OPENAI_MODEL / DASHSCOPE_API_KEY
as well as MEDICAL_QUERY_LLM_* variables.
The script invokes the configured API once and reads the medical graph.
DO NOT paste API keys into the command line.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse


def _small_anchors(anchors):
    return [{
        "surface": row.get("surface"),
        "kinds": row.get("kinds"),
        "source": row.get("source"),
        "candidate_node_count": len(row.get("node_ids") or []),
    } for row in anchors or []]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--env-file", type=Path, default=None,
                    help="Explicit .env path (optional; by default searches parent folders).")
    ap.add_argument("--question", default=(
        "布地奈德鼻喷雾剂治疗过敏性鼻炎时，能否和含伪麻黄碱的复方感冒药一起用？为什么？"
    ))
    ap.add_argument("--require-llm", action="store_true",
                    help="Fail clearly if semantic model is not available")
    args = ap.parse_args()

    if args.require_llm:
        os.environ["MEDICAL_QUERY_REQUIRE_LLM"] = "1"
    if args.env_file is not None:
        os.environ["MEDICAL_QUERY_ENV_FILE"] = str(args.env_file.resolve())
    from medical_llm_query_understanding import (
        load_query_understanding_model_from_env, medical_query_llm_config_status
    )
    import medical_query_understanding
    import medical_llm_query_understanding
    import medical_semantic_retrieval

    print("Actual imported module paths:")
    for module in (
        medical_query_understanding, medical_llm_query_understanding,
        medical_semantic_retrieval,
    ):
        print(" ", module.__name__, Path(module.__file__).resolve())
    config = medical_query_llm_config_status()
    configured = load_query_understanding_model_from_env() is not None
    print("Found .env:", config["env_file"] or "(not found)")
    print("LLM config complete:", configured)
    print("Base URL variable:", config["base_url_var"] or "(unset)")
    print("Model variable:", config["model_var"] or "(unset)")
    print("API key variable:", config["api_key_var"] or "(unset)")
    print("LLM service name:",
          os.getenv(config["model_var"], "(unset)") if config["model_var"] else "(unset)")
    print("LLM API key value: [never printed]")
    model_adapter = load_query_understanding_model_from_env()
    if model_adapter is not None:
        base_path = urlparse(model_adapter.base_url).path.rstrip("/")
        host = (urlparse(model_adapter.base_url).hostname or "").lower()
        print("API path (host hidden):", base_path or "/")
        print("Effective HTTP timeout:", model_adapter.timeout, "seconds")
        print("Enable thinking setting:", os.getenv("MEDICAL_QUERY_LLM_ENABLE_THINKING", "(model default)"))
        if host.endswith(".maas.aliyuncs.com") and "/compatible-mode/v1" not in base_path:
            print("WARNING: Alibaba Cloud workspace OpenAI-compatible API normally uses "
                  "a URL ending in /compatible-mode/v1; check your OPENAI_BASE_URL.")
    if not configured:
        print("Check the .env file path and the API key for the chosen endpoint.")

    excerpts, report = medical_semantic_retrieval.search_semantic_graph(
        args.question, project_root=args.root.resolve(),
    )
    understood = report.get("query_understanding") or {}
    summary = {
        "status": report.get("status"),
        "reason": report.get("reason"),
        "mode": report.get("query_understanding_mode"),
        "llm_call": report.get("query_understanding_llm"),
        "model_source": report.get("query_understanding_model_source")
                        or (report.get("query_understanding_llm") or {}).get("model_source"),
        "effective_model_response": (report.get("query_understanding_llm") or {}).get("response_metadata"),
        "intents": understood.get("llm_query_understanding", {}).get("intents"),
        "catalog_verified_medicines": [
            {"surface": m.get("surface"), "basis": m.get("basis"),
             "kind": "ingredient" if m.get("ingredient_ids") else "drug",
             "actual_patient_product_known": m.get("actual_patient_product_known")}
            for m in understood.get("medicines") or []
        ],
        "patient_context": [
            {"surface": x.get("surface"), "kind": x.get("kind"),
             "assertion": x.get("assertion")}
            for x in understood.get("patient_context") or []
        ],
        "unconfirmed_product_categories": understood.get("vague_medicine_details"),
        "sub_queries": report.get("semantic_sub_queries"),
        "relation_hints": report.get("semantic_query_relations"),
        "graph_primary_anchors": _small_anchors(report.get("primary_anchors")),
        "graph_soft_context_anchors": report.get("soft_context_anchors"),
        "graph_paths": report.get("returned_paths", len(excerpts)),
        "multi_relation_paths": report.get("multi_relation_paths"),
        "partial_answer_only": report.get("partial_answer_only"),
        "truncated": report.get("truncated"),
        "stats": report.get("stats"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if summary["mode"] != "llm_first":
        print("\nWARNING: 当前并未成功使用LLM语义优先，请先解决模型配置或API错误。")
        return 2
    if report.get("status") == "no_connected_evidence_path":
        print("\n提示：查询已完成语义识别，但图谱没有找到可核验的连通证据；"
              "下一步检查图谱关系与主应用的全文/向量检索回退。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
