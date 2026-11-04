"""Bounded recall, stable deduplication and one rerank for a shared query plan."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import time

# Shared cap prevents timed-out requests from creating unlimited active workers.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="medical-recall")
_RERANK_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medical-rerank")


def document_key(doc):
    meta = doc.metadata or {}
    return str(meta.get("chunk_id") or hashlib.sha256((str(meta.get("source")) + "\0" + doc.page_content).encode()).hexdigest())


def _copy(doc, extra):
    return type(doc)(page_content=doc.page_content, metadata={**doc.metadata, **extra})


def select_coverage(documents, tasks, limit):
    """Prefer relevant task/field/product coverage without inventing scores."""
    remaining = list(documents)
    selected, covered_tasks, covered_fields, covered_drugs = [], set(), set(), set()
    while remaining and len(selected) < limit:
        def gain(pair):
            rank, doc = pair
            meta = doc.metadata
            task_ids = set(meta.get("retrieval_task_ids", []))
            field = meta.get("section_field")
            drug = meta.get("drug_name")
            return (len(task_ids - covered_tasks) * 4 + int(bool(field) and field not in covered_fields) * 2
                    + int(bool(drug) and drug not in covered_drugs) * 3, -rank)
        index, doc = max(enumerate(remaining), key=gain)
        remaining.pop(index)
        selected.append(doc)
        covered_tasks.update(doc.metadata.get("retrieval_task_ids", []))
        covered_fields.add(doc.metadata.get("section_field"))
        covered_drugs.add(doc.metadata.get("drug_name"))
    return selected


def execute_retrieval_plan(plan, *, search, rerank, top_n=8, clock=time.monotonic):
    """Return available real candidates at deadline; late workers never mutate results.

    Timeout does not cancel an in-flight SDK call. Two recall workers and one
    rerank worker can remain active; adapters also set transport timeouts.
    """
    started = clock()
    tasks = plan.get("tasks", [])[:3]
    limits = plan.get("limits", {})
    recall_seconds = min(12, max(0.01, float(limits.get("recall_seconds", 12))))
    rerank_seconds = min(10, max(0.01, float(limits.get("rerank_seconds", 10))))
    total_seconds = min(24, max(0.01, float(limits.get("total_seconds", 24))))
    report = {"plan_id": plan.get("plan_id"), "planning_model_calls": plan.get("planning_model_calls"),
              "recall_calls": len(tasks), "rerank_calls": 0, "tasks": [], "timed_out": False}
    recall_end = min(started + recall_seconds, started + total_seconds)
    pending = {_POOL.submit(search, task["query"], k=12): (i, task) for i, task in enumerate(tasks)}
    groups = {}
    candidate_drugs = set(plan.get("candidate_drug_names", []))
    while pending:
        remaining = recall_end - clock()
        if remaining <= 0:
            break
        done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
        if not done:
            break
        for future in done:
            index, task = pending.pop(future)
            try:
                docs = list(future.result())[:12]
                # Typed medical rows must match the task's requested columns.
                docs = [doc for doc in docs if not doc.metadata.get("section_field")
                        or not task.get("fields") or doc.metadata["section_field"] in task["fields"]]
                docs = [doc for doc in docs if not candidate_drugs or not doc.metadata.get("drug_name")
                        or doc.metadata["drug_name"] in candidate_drugs]
                groups[index] = docs
                report["tasks"].append({"id": task["id"], "status": "ok", "candidates": len(docs)})
            except Exception as exc:
                report["tasks"].append({"id": task["id"], "status": "error", "error_type": type(exc).__name__})
    for future, (_, task) in pending.items():
        future.cancel()
        report["tasks"].append({"id": task["id"], "status": "timeout", "candidates": 0})
        report["timed_out"] = True
    report["recall_seconds"] = round(clock() - started, 4)
    pool, matches = [], {}
    # Round-robin before dedup keeps every completed query represented.
    for rank in range(max((len(d) for d in groups.values()), default=0)):
        for index in sorted(groups):
            if rank >= len(groups[index]):
                continue
            doc = groups[index][rank]
            key = document_key(doc)
            task_id = tasks[index]["id"]
            if key in matches:
                matches[key].metadata["retrieval_task_ids"].append(task_id)
            else:
                candidate = _copy(doc, {"retrieval_task_ids": [task_id]})
                matches[key] = candidate
                pool.append(candidate)
    pool = pool[:36]
    report["unique_candidates"] = len(pool)
    report["tasks"].sort(key=lambda item: next((i for i,t in enumerate(tasks) if t["id"] == item["id"]), 0))
    ranked = pool
    remaining = min(rerank_seconds, started + total_seconds - clock())
    if pool and remaining > 0:
        report["rerank_calls"] = 1
        rerank_started = clock()
        future = _RERANK_POOL.submit(rerank, plan["original_question"], pool, top_n=len(pool), timeout=remaining, max_chars=1200)
        try:
            done, _ = wait([future], timeout=remaining)
            if not done:
                raise TimeoutError("rerank budget exhausted")
            ranked = future.result()
            if (not ranked or len({document_key(d) for d in ranked}) != len(ranked)
                    or any(document_key(doc) not in matches or doc.page_content != matches[document_key(doc)].page_content for doc in ranked)):
                raise ValueError("rerank returned unknown or empty candidates")
            ranked = [_copy(doc, {"retrieval_task_ids": matches[document_key(doc)].metadata["retrieval_task_ids"]}) for doc in ranked]
            report["rerank_status"] = "ok"
        except Exception as exc:
            future.cancel()
            ranked = pool
            report["rerank_status"] = "fallback_recall_order"
            report["rerank_error_type"] = type(exc).__name__
            report["timed_out"] |= isinstance(exc, TimeoutError)
        report["rerank_seconds"] = round(clock() - rerank_started, 4)
    else:
        report["rerank_status"] = "skipped_empty" if not pool else "skipped_budget"
    selected = select_coverage(ranked, tasks, top_n)
    report["seconds"] = round(clock() - started, 4)
    report["selected"] = len(selected)
    report["coverage_scope"] = "candidate_task_coverage_not_evidence_sufficiency"
    report["uncovered_tasks"] = [task["id"] for task in tasks
                                 if not any(task["id"] in d.metadata.get("retrieval_task_ids", []) for d in selected)]
    report["scope"] = "recall_and_rerank_only_excludes_initialization_provenance_and_generation"
    return selected, report
