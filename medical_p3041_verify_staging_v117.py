# -*- coding: utf-8 -*-
"""P3-04.1 疾病标准化 Chunk 与原始 medical.json 的完整逐条回溯（只读）。

原始文件不写，staging 不写，Chroma/Neo4j 不连接，Embedding 不调用。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from medical_p3041_medical_json_prepare_v117 import (
    DATA_ORIGIN, SOURCE_STATUS, FIELD_SPECS, canonical_json, digest,
    field_text_and_stats, get_oid, iterate_source, sha256_file,
    verify_field_shape
)


def _must(test: bool, reason: str) -> None:
    if not test:
        raise ValueError(reason)


def verify(source: Path, stage: Path, manifest: Path) -> dict:
    source, stage, manifest = (
        Path(x).resolve() for x in (source, stage, manifest)
    )
    summary=json.loads(manifest.read_text(encoding="utf-8-sig"))
    _must(summary.get("stage") == "P3-04.1", "不是 P3-04.1 报告")
    _must(summary.get("status") == "STAGED_UNVERIFIED_DATA",
          "不是已完成、未触碰数据库的完整疾病标准化输出")
    raw_sha=sha256_file(source)
    _must(raw_sha == summary.get("source_file_sha256"),
          "源 medical.json SHA256 已变化")
    _must(sha256_file(stage) == summary.get("output_chunks_sha256"),
          "staging JSONL SHA256 已变化")
    originals={}
    all_fields={}
    for line_num, record, line_sha in iterate_source(source):
        verify_field_shape(record)
        oid=get_oid(record)
        _must(oid not in originals, f"重复源记录 ID: {oid}")
        originals[oid]=(line_num, line_sha, record)
        for field, section_type, _ in FIELD_SPECS:
            canon, _=field_text_and_stats(record,field)
            if canon.strip():
                all_fields[(oid,field)]=canon

    _must(len(originals)==summary.get("source_records_processed"),
          "源文件疾病记录数与标准化报告不同")
    seen_ids=set()
    coverage=defaultdict(list)
    counts=Counter()
    with stage.open("r",encoding="utf-8-sig") as stream:
        for line_num, raw in enumerate(stream,1):
            try:
                row=json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"staging 第 {line_num} 行 JSON 错误") from exc
            _must(isinstance(row,dict),f"staging 第 {line_num} 行不是对象")
            metadata=row.get("metadata")
            cid=row.get("id")
            doc=row.get("text")
            _must(isinstance(metadata,dict),f"chunk {line_num} metadata 非对象")
            _must(isinstance(cid,str) and cid not in seen_ids,
                  f"chunk 第 {line_num} 行 ID 缺失或重复")
            _must(isinstance(doc,str),f"chunk {cid} 正文不是字符串")
            seen_ids.add(cid)
            oid=metadata.get("record_oid")
            _must(oid in originals,f"chunk {cid} 未找到原始疾病 ID")
            original_line, line_sha, record=originals[oid]
            field=metadata.get("source_field")
            key=(oid,field)
            _must(key in all_fields, f"chunk {cid} 源字段为空或不存在")
            _must(metadata.get("source_line_number")==original_line,
                  f"chunk {cid} 源行号不一致")
            _must(metadata.get("source_line_sha256")==line_sha,
                  f"chunk {cid} 原始行 SHA256 不一致")
            _must(metadata.get("source_file_sha256")==raw_sha,
                  f"chunk {cid} 原始文件 SHA256 不一致")
            _must(metadata.get("source_file")==source.name,
                  f"chunk {cid} 文件名不一致")
            _must(metadata.get("disease_name")==record["name"].strip(),
                  f"chunk {cid} 疾病名称不一致")
            _must(metadata.get("disease_document_id")=="medical_json:"+oid,
                  f"chunk {cid} document_id 不一致")
            _must(metadata.get("data_origin")==DATA_ORIGIN,
                  f"chunk {cid} 来源类型不一致")
            _must(metadata.get("source_status")==SOURCE_STATUS
                  and metadata.get("clinical_verified") is False
                  and metadata.get("recommended_drug_relation") is False,
                  f"chunk {cid} 来源/医学状态被异常提升")
            _must(metadata.get("source_url")=="" and
                  metadata.get("disease_term_id")=="",
                  f"chunk {cid} 未获校验的来源 URL 或 DiseaseTerm 已被写入")
            spec=next((x for x in FIELD_SPECS if x[0]==field),None)
            _must(spec is not None and
                  metadata.get("section_type")==spec[1],
                  f"chunk {cid} 栏目类型不一致")
            canonical=all_fields[key]
            original_field=record[field]
            _must(metadata.get("source_field_sha256")==digest(
                canonical_json(original_field)),
                f"chunk {cid} 原始栏目 SHA256 不一致")
            _must(metadata.get("canonical_field_sha256")==digest(canonical),
                  f"chunk {cid} 原文基准栏目 SHA256 不一致")
            start,end=metadata.get("field_start"),metadata.get("field_end")
            _must(isinstance(start,int) and not isinstance(start,bool) and
                  isinstance(end,int) and not isinstance(end,bool) and
                  0<=start<end<=len(canonical),
                  f"chunk {cid} 原文偏移不合法")
            header=f"疾病：{record['name'].strip()}\n栏目：{spec[2]}\n"
            _must(metadata.get("document_body_start")==len(header) and
                  doc.startswith(header),f"chunk {cid} 疾病/栏目合成标题不一致")
            body=doc[len(header):]
            _must(body==canonical[start:end],
                  f"chunk {cid} 源字段字符切片不一致")
            _must(metadata.get("body_sha256")==digest(body) and
                  metadata.get("document_sha256")==digest(doc),
                  f"chunk {cid} 正文 SHA256 不一致")
            index=metadata.get("chunk_index")
            _must(isinstance(index,int) and not isinstance(index,bool)
                  and index>=0,f"chunk {cid} chunk_index 不合法")
            expect_id=(
                f"disease_v117_{oid}_{field}_{index:04d}_"
                f"{digest(body)[:12]}"
            )
            _must(cid==expect_id,f"chunk {cid} 稳定 ID 不一致")
            coverage[key].append((index,start,end,body))
            counts[f"chunks_{field}"]+=1
            counts["total"]+=1
    _must(counts["total"]==summary["index_ready_records"],
          "staging 行数与准备报告不一致")
    for (oid,field), text in all_fields.items():
        spans=sorted(coverage.get((oid,field),[]))
        _must(bool(spans),f"疾病 {oid} 栏目 {field} 没有 Chunk")
        expected_index=0
        last_end=0
        for index,start,end,body in spans:
            _must(index==expected_index and start==last_end,
                  f"疾病 {oid} 栏目 {field} Chunk 缺块/重叠/顺序错误")
            expected_index+=1
            last_end=end
        _must(last_end==len(text),
              f"疾病 {oid} 栏目 {field} 缺少尾部文本")
    for field,*_ in FIELD_SPECS:
        _must(counts[f"chunks_{field}"]==summary["field_counts"][f"chunks_{field}"],
              f"栏目 {field} 分块数量与摘要不一致")
    # 防止扫描期间原文件或 staging 被并发修改而破坏快照一致性。
    _must(sha256_file(source)==raw_sha, "校验期间源文件发生变化")
    _must(sha256_file(stage)==summary["output_chunks_sha256"],
          "校验期间 staging 发生变化")
    return {
        "stage":"P3-04.1",
        "mode":"read_only_all_chunk_source_verification",
        "status":"PASS_ALL_SOURCE_ALIGNED",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_source_records":len(originals),
        "verified_source_fields":len(all_fields),
        "verified_chunk_ids":counts["total"],
        "verified_sections":dict(sorted(
            (key,value) for key,value in counts.items() if key!="total"
        )),
        "source_file_sha256":raw_sha,
        "staging_sha256":summary["output_chunks_sha256"],
        "medical_validation":"NOT_PERFORMED",
        "publisher_and_rights_review":"NOT_PERFORMED",
        "neo4j_disease_term_matching":"NOT_PERFORMED",
        "chroma_import":"NOT_PERFORMED",
    }


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(
        description="只读校验 P3-04.1 全部疾病 Chunk 源文件 SHA、记录和字段偏移"
    )
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--chunks",type=Path,required=True)
    parser.add_argument("--prepare-report",type=Path,required=True)
    parser.add_argument("--audit-report",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.audit_report.exists():
        parser.error("审计报告文件已存在，拒绝覆盖")
    try:
        result=verify(args.input,args.chunks,args.prepare_report)
        args.audit_report.parent.mkdir(parents=True,exist_ok=True)
        with args.audit_report.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
            stream.write("\n")
    except (OSError,ValueError,RuntimeError) as exc:
        print("[P3-04.1] VERIFICATION_BLOCKED:",exc,file=sys.stderr)
        return 2
    print("[P3-04.1]",result["status"])
    print("逐条校验 Chunk:",result["verified_chunk_ids"])
    print("原始疾病记录:",result["verified_source_records"])
    print("审计报告:",args.audit_report.resolve())
    return 0


if __name__=="__main__":
    sys.exit(main())
