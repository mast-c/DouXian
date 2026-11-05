"""One local gap query, bounded verified recall, and at most one re-review."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import re
import time

from medical_evidence_coverage import evidence_bindings
from medical_query_execution import execute_retrieval_plan, RecallBatch
from medical_query_plan import FIELD_LABELS
from self_rag_retry import retry_insufficient_evidence

_REVIEW_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medical-gap-review")
_FIELD_WORDS = {"composition": r"成分|组成", "dosage": r"用法|用量|剂量|怎么用",
                "indication": r"适应症|适应证|用途|适用|治疗|改善|管用|疗效",
                "adverse_reactions": r"不良反应|副作用|不适|疼|痛",
                "contraindications": r"禁忌|禁用|不能用", "warnings": r"注意|警告|慎用",
                "drug_interactions": r"相互作用|合用|一起|联用"}


def gap_targets(targets, excerpts, coverage):
    missing = [a.get("question", "") for a in coverage.get("aspects", []) if a.get("status") == "insufficient"]
    result = []
    for target in targets:
        if any(evidence_bindings(item, [target]) for item in excerpts):
            continue
        if any(re.search(_FIELD_WORDS[target["field"]], q) and (
            target["label"] in q or len({m["surface"] for t in targets for m in t["medicines"]}) == 1
            or any(name in q for m in target["medicines"] for name in [m["surface"], *m["names"]])
        ) for q in missing):
            result.append(target)
    return result


def retry_targeted_evidence(*, model, question, targets, excerpts, coverage, original_queries,
                            search, verify, judge, recall_seconds=6, review_seconds=8):
    """Budgets bound caller waiting. Late SDK work never changes this result.

    Provenance runs inside the bounded recall worker. No rewrite model, extra
    rerank, second graph traversal or recursive retries are used.
    """
    started = time.monotonic()
    default = {"coverage": coverage, "excerpts": excerpts, "retry_queries": [], "new_chunks": 0,
               "status": "skipped", "execution": {"planning_model_calls": 0, "rerank_calls": 0}}
    if coverage.get("decision") not in ("partial", "refuse"):
        return default
    gaps = gap_targets(targets, excerpts, coverage)
    if not gaps:
        return {**default, "status": "skipped_no_source_gap"}
    # One query with a bounded scope; remaining targets are explicitly reported.
    selected_gaps = gaps[:3]
    def gap_query(rows):
        return "；".join(t["label"] for t in rows) + "。只查对应制剂这些栏目原文，保留否定及适用条件。"
    while len(selected_gaps) > 1 and len(gap_query(selected_gaps)) > 180:
        selected_gaps.pop()
    query = gap_query(selected_gaps)
    if len(query) > 180:
        return {**default, "status": "skipped_query_too_long"}
    fields = list(dict.fromkeys(t["field"] for t in selected_gaps))
    names = list(dict.fromkeys(n for t in selected_gaps for m in t["medicines"] for n in m["names"]))
    retry_plan = {"original_question": question, "planning_model_calls": 0, "candidate_drug_names": names,
                  "tasks": [{"id": "source_gap", "query": query, "fields": fields}],
                  "limits": {"recall_seconds": recall_seconds, "total_seconds": recall_seconds}}
    def recall_verified(subquery, k):
        # Filter exact product/column before spending parent-section lookups.
        candidates = [d for d in search(subquery, k=k)
                      if d.metadata.get("drug_name") in names and d.metadata.get("section_field") in fields]
        docs, report = verify(candidates, "待检索原文章节：" + "、".join(FIELD_LABELS[f] for f in fields))
        # Only return metadata across the future boundary; no late trace mutation.
        verified = [d for d in docs if d.metadata.get("evidence_type") == "json_neo4j_text_verified"
                    and (d.metadata.get("provenance_verified") or d.metadata.get("parent_verified"))]
        return RecallBatch(verified, {**report, "excluded_unverified": len(docs)-len(verified)})

    documents, execution = execute_retrieval_plan(retry_plan, search=recall_verified, rerank=None, top_n=3)
    validation = next((t.get("diagnostics", {}) for t in execution["tasks"] if t.get("status") == "ok"), {})
    execution.update({"target_ids": [t["id"] for t in selected_gaps],
                      "deferred_target_ids": [t["id"] for t in gaps[len(selected_gaps):]], "validation": validation,
                      "scope": "gap_recall_including_provenance_excludes_review"})
    if not documents:
        return {**default, "status": "retrieval_timeout" if execution["timed_out"] else "no_new_evidence",
                "retry_queries": [query], "execution": execution}
    # Keep source-gap recall trace even when re-review rejects the new candidates.
    review_budget = min(8, max(.01, review_seconds))
    if getattr(model, "root_client", None) is not None:
        client = model.root_client.with_options(timeout=review_budget, max_retries=0)
        model = model.model_copy(update={"request_timeout": review_budget, "max_retries": 0,
                                         "root_client": client, "client": client.chat.completions})

    def bounded_judge(**kwargs):
        future = _REVIEW_POOL.submit(judge, **kwargs)
        done, _ = wait([future], timeout=review_budget)
        if not done:
            future.cancel()
            execution["review_status"] = "timeout"
            execution["review_timed_out"] = True
            raise TimeoutError("source gap review budget exhausted")
        try:
            reviewed = future.result()
        except Exception as exc:
            execution["review_status"] = "error"
            execution["review_error_type"] = type(exc).__name__
            execution["review_timed_out"] = "timeout" in type(exc).__name__.lower()
            raise
        execution["review_status"] = "returned"
        return reviewed

    # The existing monotonic acceptance check retains all previously supported
    # aspects and requires the same complete checklist on the second review.
    result = retry_insufficient_evidence(
        model=model, question=question, excerpts=excerpts, coverage=coverage,
        original_queries=original_queries, search_fn=lambda q: documents,
        rewrite_fn=lambda *args: [query], judge_fn=bounded_judge,
        max_queries=1, max_new_chunks=3, max_total_chunks=len(excerpts)+3,
    )
    execution["seconds_including_review"] = round(time.monotonic()-started, 4)
    execution["review_budget_seconds"] = review_budget
    execution["review_calls"] = int(result["status"] not in ("no_queries", "no_new_evidence", "skipped"))
    execution["candidate_new_chunks"] = result["new_chunks"]
    execution["accepted_new_chunks"] = max(0, len(result["excerpts"]) - len(excerpts))
    return {**result, "new_chunks": execution["accepted_new_chunks"], "execution": execution}
