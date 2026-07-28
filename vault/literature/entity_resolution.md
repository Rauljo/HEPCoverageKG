# Literature Review: Entity Deduplication and Canonicalization in Scientific Knowledge Graphs

*Compiled collaboratively via Antigravity Research and Elicit AI (2026-07-28)*

### Executive Summary
Entity resolution (ER) in scientific Knowledge Graphs (KGs) has shifted from purely rule-based or probabilistic string matching to **Hybrid Semantic/Graph-Native Pipelines**. A recent literature search surfaced a highly consistent pattern: in recent technical KG work, deduplication is managed through hybrid pipelines where embeddings or LLMs propose candidates, and a harder layer of ontology, relation, or graph-structure checks vetoes bad merges.

The field is converging on a **three-layer design**:
1. Deterministic normalization for the truly exact parts of the string.
2. Candidate generation with embeddings or LLMs.
3. A final validator built from ontology constraints, relation constraints, or graph-neighborhood evidence.

---

### 1. Specific Frameworks & Papers
- **Wikontic (Chepurova et al., 2026)**: Triple extraction with LLMs followed by ontology-based typing, schema validation, entity deduplication, alias tracking, and automatic enforcement of Wikidata's ontology. Provides a clean extraction-plus-verification stack.
- **Prompt Me One More Time**: A two-step pipeline that extracts entity candidates, refines/links them to canonical entities, and applies relation constraints to keep only verified triplets.
- **EchoLLM**: Extracts candidate triples with an instruction-following LLM, retrieves sentence evidence with BM25 + dense retrieval, and uses Natural Language Inference (NLI) so only entailed, lexically consistent hypotheses are accepted.
- **Biomedical/Therapeutics (TheraPy)**: Aligns therapeutics using resource-provided cross-references and active-ingredient annotations rather than just string similarity. Another paper uses a generate-and-rank concept normalizer via a candidate generator plus a BERT list-wise ranker with a semantic-type regularizer.
- **Graph-Heavy Counterparts (GEBERT & EAGER)**: GEBERT learns from UMLS graph structure with graph neural networks over local neighborhoods. EAGER combines graph embeddings with attribute values in supervised ER, working better on deeper KGs.
- **Graph-Topology Avoids Lexical Brittleness (SeMBlock & Geo-ER)**: Multi-KG matching fails due to incorrect transitive identity links (exactly the "tiny surface change" failure mode). SeMBlock uses LSH over word embeddings to build a weighted graph, then prunes edges for blocking. Geo-ER uses surrounding entities to make context-based predictions, overriding brittle string similarity.

---

### 2. Handling Precise Technical Variations
The literature confirms that the "13 TeV vs 14 TeV" or "2.2.1 vs 2.2.2" problem is handled indirectly but aggressively through **semantic-type guards, alias tracking, relation constraints, or neighborhood-based scoring** rather than a dedicated generic parser. 

How this maps to our architecture:
*   **Targeted Attribute Extraction / Deterministic Guards**: Before semantic comparison, unstructured strings must be subjected to hard, rule-based blocks on specific entity types (e.g. if a version number or precise measurement magnitude differs, the match is vetoed).
*   **Contextual Disambiguation via Edges**: "13 TeV" and "14 TeV" are separated by graph-topology because they link to distinct collider runs, dates, and resulting publications. Graph-based ER uses this disjoint neighborhood to keep the entities separated.

---

### 3. Translation to HEPCoverageKG Alias Pipeline
This synthesizes perfectly into the strategy established in `D-025`. The alias layer will be a dedicated, offline batch script that performs:
1.  **Tier 1 (High-Recall Blocking)**: Fast vector embeddings (`bge-small-en-v1.5`) + n-gram blocking to group candidates, strictly within the same entity `kind`.
2.  **Tier 2 (Domain-Specific Heuristics)**: Hard deterministic regex rules for critical scientific attributes (exact matching required for version numbers, energies, chemical elements). Vetoes bad merges instantly.
3.  **Tier 3 (LLM & Graph Reasoning)**: An LLM-in-the-loop evaluator that analyzes the topological neighborhood (which papers these entities appear in, what they connect to) of the remaining candidate pairs and outputs a match decision with a generated explanation for human confirmation.
