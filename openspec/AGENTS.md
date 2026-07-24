# openspec

## Purpose

Owns the current capability specifications and proposed changes for SATI.

## Ownership

- `specs/` records implemented behavior.
- `changes/` records proposed deltas and implementation plans.
- `config.yaml` supplies project context and artifact-writing rules.

## Local Contracts

- Current specs describe shipped behavior, even when a proposed change will replace it.
- Change specs use explicit `ADDED`, `MODIFIED`, `REMOVED`, and `RENAMED` sections.
- A change must keep proposal capability names synchronized with its delta-spec paths.
- Breaking development-stage changes require no compatibility layer unless a change says otherwise.

## Work Guidance

- Source current behavior from code, tests, `DESIGN-DOC-v1.md`, and `REPORTS/`.
- Treat `DESIGN-DOC-v2.md` and `DESIGN-DOC-SYNTH.md` as forward constraints, not current capabilities.
- Validate changes with `openspec validate <name> --type change --strict`.

## Verification

- `openspec validate --all --strict`

## Child DOX Index

(none)
