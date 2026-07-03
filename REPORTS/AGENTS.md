# REPORTS — durable work reports

## Purpose

Durable implementation and experiment reports for project work that needs a
continuous record of steps, decisions, verification, and findings. Structure
mirrors dojo's `REPORTS/`.

## Ownership

Owns human-readable Markdown reports under `REPORTS/`. Reports may summarize
commands, outputs, decisions, and links to produced artifacts, but they do not
own runtime code contracts.

## Local Contracts

- Keep reports chronological and factual.
- **Every entry heading carries a timestamp: `## YYYY-MM-DD HH:MM TZ — Title`**
  (Sidney's rule, 2026-07-02; also applies to `QUESTIONS-FOR-SIDNEY.md` entries).
- Record decision junctions in `QUESTIONS-FOR-SIDNEY.md`; link or mention them
  from the report when relevant.
- Do not store generated artifacts, Parquet results, or large figures in
  `REPORTS/`; link to run directories instead.

## Work Guidance

- Prefer concise timestamped entries with command intent, result, and next
  action. Include test counts in verification lines.

## Verification

- Reports are verified by review and by the referenced command/test outcomes.

## Child DOX Index

(none)
