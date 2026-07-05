# HEPCoverageKG

A typed knowledge graph mapping which High Energy Physics final states have been
measured, by which experiment, at which energy - built to surface coverage gaps
for BSM physics models.

Pipeline: InspireHEP -> arXiv -> HEPData harvesting, RAG-based structured field
extraction, and storage in a SQLite + NetworkX knowledge graph (Neo4j later).

Reference architecture ported from DeepCollector.
