# src/sati/config_loader — compose → validate → resolve

## Purpose

Turns CLI selectors/overrides into a validated, path-resolved `RootConfig`. Structure mirrors dojo's `config_loader` (per Sidney's preference).

- `compositor.py` — Hydra Compose API (no `@hydra.main`, no multirun), searchpath fallthrough, `--resolved-config` loading.
- `resolver.py` — `run_id`, type-driven `SatiPath` absolutization, derived values (`model.imgsz`), writes `resolved.yaml`.
- `conductor.py` — sequences compose → validate → resolve into `compose_and_resolve`, the single entry point for CLI/tests/orchestrators.

## Ownership

Owns composition and resolution. Does not own the schema (`config_schemas/`), the packaged YAML (`config_defaults/`), or hashing (`sati/hashing.py`).

## Local Contracts

- Config dir precedence: explicit `--config-dir` → local `./config_defaults` → packaged tree, joined by `hydra.searchpath`; the dir owning the primary config is reordered first (Hydra requires the primary in the main dir, and group lookups search main-dir-first).
- Hybrid root selection: an `experiment=name` override makes `experiment/name` the primary config (a full root with its own defaults list, dojo-style); without it, `config.yaml` is the root. A local experiment file therefore shadows/extends packaged groups with no local `config.yaml`; group-fragment shadowing without an experiment still requires a local `config.yaml`.
- All Hydra usage ends inside `compositor.py`; downstream code sees only typed `RootConfig`.
- `--resolved-config` keeps values verbatim, applies only passed overrides, and gets a fresh `run_id` (derived-only, excluded from `config_hash`).

## Verification

- `.venv/bin/python -m pytest tests/test_config_loader.py tests/test_hashing.py`
