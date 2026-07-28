#!/usr/bin/env bash
# =============================================================================
# HEPCoverageKG: Neo4j Bulk Import Script
#
# This script uses neo4j-admin to construct a brand new local Neo4j database
# directly from the exported CSV files. Because we used the bulk-import header
# format (e.g. :START_ID, :END_ID, :TYPE, :LABEL), this is much faster and
# simpler than writing LOAD CSV cypher queries for every relationship type.
# =============================================================================

set -e

# Default to the current directory if not specified
GRAPH_DIR="${1:-data/processed/graph}"
DB_NAME="neo4j"

if [ ! -d "$GRAPH_DIR" ]; then
    echo "Error: Graph directory '$GRAPH_DIR' not found."
    echo "Please run 'hepcoveragekg graph export $GRAPH_DIR' first."
    exit 1
fi

echo "Importing CSVs from $GRAPH_DIR into Neo4j database '$DB_NAME'..."

# The neo4j-admin command. 
# You may need to prefix this with 'bin/' or run it from your Neo4j home directory,
# depending on how Neo4j Community is installed on your machine.
neo4j-admin database import full "$DB_NAME" \
    --nodes="$GRAPH_DIR/nodes_paper.csv" \
    --nodes="$GRAPH_DIR/nodes_occurrence.csv" \
    --nodes="$GRAPH_DIR/nodes_canonical.csv" \
    --nodes="$GRAPH_DIR/nodes_literal.csv" \
    --relationships="$GRAPH_DIR/edges_has_occurrence.csv" \
    --relationships="$GRAPH_DIR/edges_resolves_to.csv" \
    --relationships="$GRAPH_DIR/edges_assertion.csv" \
    --multiline-fields=true \
    --overwrite-destination=true

echo ""
echo "Import complete!"
echo "To use this database, ensure Neo4j is stopped, then run:"
echo "  neo4j start"
echo "Then open your browser to http://localhost:7474 and connect to the default 'neo4j' database."
