# src/sati/config_loader — compose → validate → resolve

## Purpose

Turns CLI selectors/overrides into a validated, path-resolved `RootConfig`. Structure mirrors dojo's `config_loader` (per Sidney's preference).

- `compositor.py` — Hydra Compose API (no `@hydra.main`, no multirun), searchpath fallthrough, `--resolved-config` loading.
- `resolver.py` — command context, identities, templates, type-driven `SatiPath` resolution, and derived values.
- `conductor.py` — side-effect-free compose → validate → resolve and provenance context.

## Ownership

Owns composition and resolution. Does not own the schema (`config_schemas/`), the packaged YAML (`config_defaults/`), or hashing (`sati/hashing.py`).

## Local Contracts

- Config dir precedence: repeatable explicit `--config-dir` → automatic project `./configs` → packaged defaults; the primary-owning directory is reordered first for Hydra.
- `experiment=name`, authored `--config`, and exact `--resolved-config` are exclusive roots. Selector-less composition uses packaged `config.yaml`.
- All Hydra usage ends inside `compositor.py`; downstream code sees only typed `RootConfig`.
- `--resolved-config` revalidates and preserves command, IDs, hashes, and paths verbatim and rejects behavioral overrides.
- Resolution does not write. Lifecycle code persists `config/resolved.yaml` as worker authority after collision validation.

## Verification

- `nix develop -c .venv/bin/python -m pytest tests/test_config_loader.py tests/test_hashing.py tests/test_checkpoints_lifecycle.py`
