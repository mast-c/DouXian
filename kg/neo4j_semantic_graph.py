"""Add verified local semantic and dataset-category graphs to Neo4j.

The source SQLite indexes and their source manifests are read-only. Writes are
additive, idempotent, and require the explicit ``apply --apply`` command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

from medical_semantic_graph import GraphBlocked, digest, fingerprint, open_graph

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DEFAULT = "semantic_graph/medical_graph.sqlite"
CATEGORY_DEFAULT = "semantic_graph/dataset_categories.sqlite"
NODE_KINDS = {
    "Drug", "Ingredient", "DrugClass", "Disease", "Symptom", "Population",
    "Condition", "Mechanism", "MechanismStatement", "DatasetCategory",
    "CatalogDrug", "DiseaseCategory",
}
RELATIONS = {
    "CLASS_MEMBERSHIP_ASSERTION", "HAS_SYMPTOM", "DISEASE_CATEGORY_MENTION", "HAS_INGREDIENT",
    "INDICATION_MENTION", "CONTRAINDICATION_MENTION", "CAUTION_MENTION",
    "INTERACTION_MENTION", "ADVERSE_REACTION_MENTION", "MECHANISM_MENTION",
    "MECHANISM_STATEMENT_MENTION", "MECHANISM_PARTNER_MENTION",
    "MECHANISM_ENTITY_MENTION", "METABOLISM_MENTION",
    "ENZYME_MODULATION_MENTION", "EXPOSURE_CHANGE_MENTION",
    "TRANSPORT_MENTION", "MECHANISM_OUTCOME_MENTION",
    "STATEMENT_ENZYME_MENTION", "STATEMENT_PROCESS_MENTION",
    "STATEMENT_EXPOSURE_MENTION", "STATEMENT_OUTCOME_MENTION",
    "DATASET_CATEGORY_MEMBERSHIP", "DATASET_CATEGORY_PARENT",
}


def _category_index(path: Path) -> tuple[sqlite3.Connection, dict]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise GraphBlocked("Dataset category graph or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != "dataset-category-1"
            or fingerprint(path) != manifest.get("index")):
        raise GraphBlocked("Dataset category graph index changed; rebuild required")
    source_root = ROOT / "data/raw/drug_db/MedGraphRAG-Drug-DB"
    for name, expected in manifest.get("files", {}).items():
        if name not in {"drug_categories_western.json", "drug_categories_chinese.json"}:
            raise GraphBlocked("Unexpected category source file")
        if fingerprint(source_root / name) != expected:
            raise GraphBlocked("Category source changed; rebuild required: " + name)
    conn = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn, manifest


def inspect(semantic_path: Path, category_path: Path) -> dict:
    semantic_path = semantic_path.resolve()
    category_path = category_path.resolve()
    sem, semantic_manifest = open_graph(ROOT, semantic_path)
    cat = None
    try:
        kinds = dict(sem.execute("SELECT kind,count(*) FROM entities GROUP BY kind"))
        relations = dict(sem.execute("SELECT relation,count(*) FROM edges GROUP BY relation"))
        invalid = sem.execute(
            "SELECT count(*) FROM entities e WHERE e.kind NOT IN (" +
            ",".join("?" for _ in NODE_KINDS) + ")", tuple(NODE_KINDS)
        ).fetchone()[0]
        invalid_relations = [r[0] for r in sem.execute(
            "SELECT DISTINCT relation FROM edges ORDER BY relation") if r[0] not in RELATIONS]
        if invalid or invalid_relations:
            raise GraphBlocked("Semantic graph contains unsupported labels or relations")
        cat, category_manifest = _category_index(category_path)
        category_kinds = dict(cat.execute("SELECT kind,count(*) FROM entities GROUP BY kind"))
        category_relations = dict(cat.execute("SELECT relation,count(*) FROM edges GROUP BY relation"))
        bad_endpoints = cat.execute(
            "SELECT count(*) FROM edges e LEFT JOIN entities a ON a.id=e.src "
            "LEFT JOIN entities b ON b.id=e.dst WHERE a.id IS NULL OR b.id IS NULL"
        ).fetchone()[0]
        if bad_endpoints:
            raise GraphBlocked("Dataset category graph has missing endpoints")
        return {
            "status": "LOCAL_PREFLIGHT_OK",
            "semantic_graph": {"path": str(semantic_path), "snapshot": semantic_manifest["index"]["sha256"],
                               "nodes": semantic_manifest["nodes"], "edges": semantic_manifest["edges"],
                               "sources": semantic_manifest["sources"], "node_kinds": kinds,
                               "relations": relations},
            "category_graph": {"path": str(category_path), "snapshot": category_manifest["signature"],
                               "nodes": sum(category_kinds.values()),
                               "edges": sum(category_relations.values()),
                               "node_kinds": category_kinds, "relations": category_relations,
                               "clinical_validation": False},
            "planned_node_rows": semantic_manifest["nodes"] + sum(category_kinds.values()),
            "planned_relationship_rows": semantic_manifest["edges"] + sum(category_relations.values()),
        }
    finally:
        sem.close()
        if cat is not None:
            cat.close()


def _rows(conn, query, batch_size=500):
    cursor = conn.execute(query)
    while rows := cursor.fetchmany(batch_size):
        yield [dict(row) for row in rows]


def _neo4j_connect(uri: str | None, user: str | None, database: str | None):
    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD 未设置；未连接数据库")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("缺少 Neo4j 驱动，请安装 neo4j Python package") from exc
    driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
    driver.verify_connectivity()
    return driver, database


def _node_query(kind: str, snapshot_field: str) -> str:
    if snapshot_field not in {"semantic_graph_snapshot", "dataset_category_snapshot"}:
        raise ValueError("Unsupported graph snapshot property")
    return f"""UNWIND $rows AS row
MERGE (n:{kind} {{id:row.id}})
SET n.name=row.name,
    n.{snapshot_field}=$snapshot
"""


def _edge_query(relation: str, src_kind: str, dst_kind: str) -> str:
    if src_kind not in NODE_KINDS or dst_kind not in NODE_KINDS:
        raise ValueError("Unsupported relationship endpoint kind")
    return f"""UNWIND $rows AS row
MATCH (a:{src_kind} {{id:row.src}})
MATCH (b:{dst_kind} {{id:row.dst}})
MERGE (a)-[r:{relation} {{id:row.id}}]->(b)
SET r += row.properties
"""


def _validate_existing_nodes(session, rows, kind):
    query = f"""UNWIND $rows AS row
OPTIONAL MATCH (n:{kind} {{id:row.id}})
RETURN row.id AS id, row.name AS expected_name, n.name AS actual_name
"""
    result = session.run(query, rows=rows)
    conflicts = [r["id"] for r in result
                 if r["actual_name"] is not None and r["actual_name"] != r["expected_name"]]
    if conflicts:
        raise GraphBlocked(f"Neo4j {kind} 节点 ID 对应名称不一致，例如 {conflicts[0]}")


def _import_graphs(session, semantic_path: Path, category_path: Path, batch_size: int):
    sem, semantic_manifest = open_graph(ROOT, semantic_path)
    cat, category_manifest = _category_index(category_path)
    semantic_snapshot = semantic_manifest["index"]["sha256"]
    category_snapshot = category_manifest["signature"]
    imported = {"nodes": 0, "relationships": 0}
    try:
        node_sets = [(sem, "SELECT id,kind,name FROM entities ORDER BY kind,id", "semantic_graph_snapshot", semantic_snapshot),
                     (cat, "SELECT id,kind,name FROM entities ORDER BY kind,id", "dataset_category_snapshot", category_snapshot)]
        # Check every identity before the first write, so a late name collision
        # cannot leave a partially imported graph.
        for conn, query, _, _ in node_sets:
            for rows in _rows(conn, query, batch_size):
                by_kind = {}
                for row in rows:
                    if row["kind"] not in NODE_KINDS:
                        raise GraphBlocked("Unsupported entity kind: " + row["kind"])
                    by_kind.setdefault(row["kind"], []).append({"id": row["id"], "name": row["name"]})
                for kind, batch in by_kind.items():
                    _validate_existing_nodes(session, batch, kind)
        for kind in sorted(NODE_KINDS):
            safe_kind = kind.lower()
            session.run(
                f"CREATE CONSTRAINT semantic_{safe_kind}_id IF NOT EXISTS "
                f"FOR (n:{kind}) REQUIRE n.id IS UNIQUE"
            ).consume()
        # Drug IDs overlap the existing base graph and must have the same name.
        for conn, query, snapshot_field, snapshot in node_sets:
            for rows in _rows(conn, query, batch_size):
                by_kind = {}
                for row in rows:
                    by_kind.setdefault(row["kind"], []).append({"id": row["id"], "name": row["name"]})
                for kind, batch in by_kind.items():
                    session.run(_node_query(kind, snapshot_field), rows=batch,
                                snapshot=snapshot).consume()
                    imported["nodes"] += len(batch)
        # Section evidence is kept on each edge; no clinical relation is inferred.
        edge_query = """SELECT e.id,e.src,e.dst,src.kind AS src_kind,dst.kind AS dst_kind,e.relation,e.source_id,e.quote_start,e.quote_end,
e.mention_start,e.mention_end,e.qualifier_json,e.status,s.record_id,s.field,s.url,s.authority,
s.text_sha256,s.file,s.byte_start,s.byte_length,s.row_sha256,s.selector,s.text
FROM edges e JOIN sources s ON s.id=e.source_id
JOIN entities src ON src.id=e.src JOIN entities dst ON dst.id=e.dst
ORDER BY e.relation,e.id"""
        for rows in _rows(sem, edge_query, batch_size):
            by_relation = {}
            for row in rows:
                relation = row.pop("relation")
                src_kind, dst_kind = row.pop("src_kind"), row.pop("dst_kind")
                if relation not in RELATIONS:
                    raise GraphBlocked("Unsupported relation: " + relation)
                quote = row.pop("text")[row["quote_start"]:row["quote_end"]]
                row["properties"] = {
                    **row,
                    "quote": quote,
                    "graph_snapshot": semantic_snapshot,
                }
                key = (relation, src_kind, dst_kind)
                by_relation.setdefault(key, []).append({"id": row["id"], "src": row["src"],
                                                          "dst": row["dst"], "properties": row["properties"]})
            for (relation, src_kind, dst_kind), batch in by_relation.items():
                session.run(_edge_query(relation, src_kind, dst_kind), rows=batch).consume()
                imported["relationships"] += len(batch)
        category_edge_query = """SELECT e.id,e.src,e.dst,src.kind AS src_kind,dst.kind AS dst_kind,
e.relation,e.file,e.json_pointer,e.source_sha256
FROM edges e JOIN entities src ON src.id=e.src JOIN entities dst ON dst.id=e.dst
ORDER BY e.relation,e.id"""
        for rows in _rows(cat, category_edge_query, batch_size):
            by_relation = {}
            for row in rows:
                relation = row.pop("relation")
                src_kind, dst_kind = row.pop("src_kind"), row.pop("dst_kind")
                if relation not in RELATIONS:
                    raise GraphBlocked("Unsupported category relation: " + relation)
                row["properties"] = {**row, "graph_snapshot": category_snapshot,
                                     "source_status": "local_dataset_unverified",
                                     "clinical_validation": False}
                key = (relation, src_kind, dst_kind)
                by_relation.setdefault(key, []).append({"id": row["id"], "src": row["src"],
                                                          "dst": row["dst"], "properties": row["properties"]})
            for (relation, src_kind, dst_kind), batch in by_relation.items():
                session.run(_edge_query(relation, src_kind, dst_kind), rows=batch).consume()
                imported["relationships"] += len(batch)
        return {**imported, "semantic_snapshot": semantic_snapshot,
                "category_snapshot": category_snapshot}
    finally:
        sem.close()
        cat.close()


def verify_database(session, local: dict) -> dict:
    semantic = local["semantic_graph"]
    category = local["category_graph"]
    problems = []
    node_counts = {}
    for kind, expected in semantic["node_kinds"].items():
        actual = session.run(
            f"MATCH (n:{kind} {{semantic_graph_snapshot:$snapshot}}) RETURN count(n) AS count",
            snapshot=semantic["snapshot"],
        ).single()["count"]
        node_counts["semantic:" + kind] = {"expected": expected, "actual": actual}
        if actual != expected:
            problems.append(f"semantic {kind}: {actual} != {expected}")
    for kind, expected in category["node_kinds"].items():
        actual = session.run(
            f"MATCH (n:{kind} {{dataset_category_snapshot:$snapshot}}) RETURN count(n) AS count",
            snapshot=category["snapshot"],
        ).single()["count"]
        node_counts["category:" + kind] = {"expected": expected, "actual": actual}
        if actual != expected:
            problems.append(f"category {kind}: {actual} != {expected}")

    edge_counts = {}
    expected_relations = {}
    for relation, count in semantic["relations"].items():
        expected_relations[relation] = expected_relations.get(relation, 0) + count
    for relation, count in category["relations"].items():
        expected_relations[relation] = expected_relations.get(relation, 0) + count
    for relation, expected in expected_relations.items():
        actual = 0
        for snapshot in (semantic["snapshot"], category["snapshot"]):
            actual += session.run(
                f"MATCH ()-[r:{relation} {{graph_snapshot:$snapshot}}]->() RETURN count(r) AS count",
                snapshot=snapshot,
            ).single()["count"]
        edge_counts[relation] = {"expected": expected, "actual": actual}
        if actual != expected:
            problems.append(f"{relation}: {actual} != {expected}")
    return {"status": "VERIFIED" if not problems else "FAILED",
            "nodes": node_counts, "relationships": edge_counts, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "apply", "verify"))
    parser.add_argument("--semantic-index", default=SEMANTIC_DEFAULT)
    parser.add_argument("--category-index", default=CATEGORY_DEFAULT)
    parser.add_argument("--uri")
    parser.add_argument("--user")
    parser.add_argument("--database")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--apply", action="store_true", help="required to write Neo4j")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 5000:
        parser.error("batch-size must be between 1 and 5000")
    if args.command != "apply" and args.apply:
        parser.error("--apply is only valid with the apply command")
    try:
        local = inspect(ROOT / args.semantic_index, ROOT / args.category_index)
        if args.command == "preflight":
            result = local
        else:
            if args.command == "apply" and not args.apply:
                raise RuntimeError("写入命令必须同时提供 --apply")
            driver, database = _neo4j_connect(args.uri, args.user, args.database)
            try:
                with driver.session(database=database) as session:
                    if args.command == "apply":
                        result = {**local, "import": _import_graphs(
                            session, ROOT / args.semantic_index, ROOT / args.category_index,
                            args.batch_size)}
                    else:
                        result = {**local, "database_check": verify_database(
                            session, local)}
            finally:
                driver.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "verify" and result["database_check"]["status"] != "VERIFIED":
            sys.exit(1)
    except (OSError, RuntimeError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
