# HEPCoverageKG

Typed knowledge graph for HEP coverage mapping (MSc dissertation, UCL).
All project context, decisions, and history live in `vault/` — read it before working.

## Session conventions (vault)

At session start:
1. Read `vault/overview.md` and the most recent file in `vault/logs/`.
2. Read `vault/ideas/index.md` (status line + list).
3. Check `vault/inbox/` — integrate anything found into the proper vault files, then delete the raw file.

During the session (incremental, never end-loaded):
- Decision made → append a numbered `D-nnn` entry to `vault/decisions.md` immediately.
- Idea discussed → create/update its doc in `vault/ideas/` and refresh the status-counts line at the top of `vault/ideas/index.md`.
- Paper/system discussed → add or extend an entry in `vault/literature.md`.

At session end (best effort):
- Write `vault/logs/YYYY-MM-DD.md` (short: happened / decided / open / next).
- Refresh `vault/overview.md` if project state changed.

Git: vault changes are committed separately from code, message prefix `vault:`. Commit only when the user asks.

Reference codebase (read-only, port patterns, never edit):
`/Users/raulsal/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Dissertation/HEPKG`
