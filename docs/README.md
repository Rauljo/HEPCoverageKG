# Schema diagrams

Entity-relationship diagrams of the milestone-1 bundle-import store, generated
from [`hepcoveragekg/kg/schema.sql`](../hepcoveragekg/kg/schema.sql).

| Diagram | Scope |
|---|---|
| [`schema-er-full`](schema-er-full.svg) | All 15 tables and their enforced foreign keys (provenance, entities, scientific edges, QA/review metadata, status history). |
| [`schema-er-physics`](schema-er-physics.svg) | The **scientific graph only** — entities (nodes), assertions (subject→object edges) with lifecycle status, and backing evidence. This is the subgraph projected to NetworkX/Neo4j. |

Each is provided as `.svg` (vector, best for the write-up) and `.png` (2× raster).
Crow's-foot lines are enforced SQLite foreign keys. Composite-key columns are
marked `PK`; the FK direction is shown by the relationship lines. The two soft
pointers (`paper.latest_bundle_id`, `paper.source_hash`, no FK) are omitted.

## Regenerating after a schema change

Edit the `.mmd` source, then render with mermaid-cli (needs Node):

```sh
cd docs
npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -i schema-er-full.mmd    -o schema-er-full.svg    -b white
npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -i schema-er-full.mmd    -o schema-er-full.png    -b white -s 2
npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -i schema-er-physics.mmd -o schema-er-physics.svg -b white
npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -i schema-er-physics.mmd -o schema-er-physics.png -b white -s 2
```

(`.puppeteer.json` just passes `--no-sandbox`.) Note: keep column key markers to
`PK`/`FK`/`UK` — mermaid's ER parser rejects combined tokens like `PK_FK`.
