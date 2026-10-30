// Open http://localhost:7475 (dedicated DrugKG Neo4j).
// Default login: neo4j ; password is the value you set locally.

// 1. Data volumes: expect 11953 drugs, 11960 records,
// 106445 sections, 107787 chunks, 343 alias candidates,
// 1150 disease terms, 19685 candidate mentions, 60 model assessments.
MATCH (n:Drug) RETURN count(n) AS drugs;

// Run these separately
MATCH (n:SourceRecord) RETURN count(n) AS source_records;
MATCH (n:Section) RETURN count(n) AS sections;
MATCH (n:Chunk) RETURN count(n) AS chunks;
MATCH (n:DrugAlias) RETURN count(n) AS candidate_aliases;
MATCH (n:DiseaseTerm) RETURN count(n) AS disease_terms;
MATCH (n:DiseaseCandidate) RETURN count(n) AS disease_candidates;
MATCH (n:ModelAssessment) RETURN count(n) AS llm_assessments;

// 2. Drug -> disease *candidate* -> exact evidence and source.
// This is NOT a clinically validated therapeutic relationship.
MATCH (drug:Drug)-[:HAS_CANDIDATE]->(candidate:DiseaseCandidate)
      -[:MENTIONS_TERM]->(term:DiseaseTerm)
MATCH (candidate)-[:FROM_RECORD]->(src:SourceRecord)
WHERE drug.name CONTAINS '左氧氟沙星'
RETURN drug.name AS drug,
       term.name AS matched_term,
       candidate.context_flag AS context_flag,
       candidate.evidence_quote AS evidence_quote,
       candidate.candidate_status AS review_status,
       src.source_url AS source_url
LIMIT 30;

// 3. Disease -> drugs (candidate mentions only)
MATCH (term:DiseaseTerm {name:'心绞痛'})
      <-[:MENTIONS_TERM]-(candidate:DiseaseCandidate)
      <-[:HAS_CANDIDATE]-(drug:Drug)
RETURN drug.name AS drug,
       candidate.evidence_quote AS source_quote,
       candidate.clinical_status AS clinical_status
LIMIT 40;

// 4. Model assessment (60 dev labels remain separate from clinical facts)
MATCH (a:ModelAssessment)-[:ASSESSES]->(c:DiseaseCandidate)
MATCH (d:Drug)-[:HAS_CANDIDATE]->(c)
RETURN d.name AS drug, c.disease_name AS candidate_term,
       a.candidate_decision AS llm_proposal,
       a.text_relation_status AS conditional_or_specificity,
       a.review_status AS assessment_status
LIMIT 30;

// 5. Graph path navigation: drug -> source -> section -> chunk
MATCH p=(d:Drug)-[:HAS_RECORD]->(:SourceRecord)
        -[:HAS_SECTION]->(:Section {field:'indication'})
        -[:HAS_CHUNK]->(:Chunk)
RETURN p LIMIT 5;

// All candidate edges are mentions, not clinical indications.
// Do NOT infer d-[:TREATS]->t or d-[:INDICATED_FOR]->t from this data.
