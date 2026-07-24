# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## Project Contracts

- Implementation follows `DESIGN-DOC-v1.md` first; `DESIGN-DOC-v2.md` is a delta applied later; `DESIGN-DOC-SYNTH.md` governs `src/synth_dataset_builder`. (The original `DESIGN-DOC.md` was superseded and has been deleted.)
- Open design decisions are marked with `> [!NOTE]` callouts in the design docs; settled decisions are folded into spec prose. Do not re-open settled items.
- Development environment is the repo-local `.venv` (Python 3.14). All installs, tests, and scripts run through `.venv/bin/python`; the shared `pyproject.toml` is the single dependency source (`pip install -e ".[dev,synth]"`; `[ultralytics]` extra deferred until training work starts).
- Coordinate convention everywhere: half-open `xyxy`, top-left origin.

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

- No backwards compatibility across dev stages: v1 artifacts need not survive into v2, synth stage N artifacts need not survive into stage N+1.
- Non-blocking questions and deferred decisions go in `QUESTIONS-FOR-SIDNEY.md` at repo root for later review; only stop work for genuinely big blockers.
- Where SATI shares concepts with dojo (`~/Projects/image-classifier-dojo`) — CLI grammar, config composition, schema packages, input methodology, ID/hash semantics, output templates, and run/training artifacts — use the same design language and structure while preserving SATI's domain-specific commands and results.
- Durable work reports live in `REPORTS/` (dojo-style continuous record). Every report entry and every `QUESTIONS-FOR-SIDNEY.md` entry carries a timestamp `YYYY-MM-DD HH:MM TZ` (Sidney, 2026-07-02).

## Child DOX Index

- `configs/AGENTS.md` — authored local Hydra experiment configs and supporting config inputs
- `src/sati/AGENTS.md` — the SATI package (prepare/convert/train/eval/infer/export/inspect per DESIGN-DOC-v1)
- `src/synth_dataset_builder/AGENTS.md` — standalone synthetic dataset builder / SATI stress suite (per DESIGN-DOC-SYNTH)
- `REPORTS/AGENTS.md` — durable timestamped implementation/experiment reports
- `openspec/AGENTS.md` — current capability specs and proposed change artifacts

Owned by root: design docs, `pyproject.toml`, `QUESTIONS-FOR-SIDNEY.md`, top-level `tests/`.
