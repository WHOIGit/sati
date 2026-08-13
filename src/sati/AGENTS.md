# src/sati

## Purpose

The SATI package: slice-aware prepare/convert/train/eval/infer/export plus a read-only `inspect` sub-app, per `DESIGN-DOC-v1.md` (v2 delta comes later).

## Ownership

Owns all `sati.*` modules, including strict schemas/config defaults, side-effect-free composition/resolution, command-aware hashing and dataset identity, collision-safe lifecycles, checkpoint contracts, canonical manifests/results, slicing/recombination, stock-Ultralytics backends, command workers, inspection, and CLI.

## Local Contracts

- Pydantic v2 schemas are the single source of truth: `extra="forbid"`, absent = unsupported. Validation once at the boundary; downstream code takes typed models, never `DictConfig`.
- All path-typed schema fields use the `SatiPath` annotated alias — bare `pathlib.Path` in a schema is a bug (config_hash strips by annotation; DESIGN-DOC-v1 §4.7).
- Slicing supports `level_policy.kind=auto` and `start_then_geometric`; **each policy owns its own factor** (`level_policy.downsample_factor` / `.subsequent_downsample_factor` — no shared top-level knob). Level index 0 ⇔ full-res always; below-full-res stacks number from 1, so pyramids are keyed by level index (`LevelPyramid`), never list position.
- Stretched slice windows are emitted via the sharp-source rule (finest sufficient level, else full-res; §6.1) — never upscaled from their own level's pixels.
- Split TOML entries may name source image filenames, source image stems, or SATI fullframe ids; aliases must resolve unambiguously.
- `training.yolo_augment.*` fields are optional pass-throughs to stock Ultralytics train kwargs; omitted fields preserve Ultralytics defaults.
- Packaged Hydra defaults resolve via `importlib.resources.files("sati")` to a real directory; hard error on zip imports (§3.2). Compose API only, never `@hydra.main`.
- Config search is explicit `--config-dir` → automatic `./configs` → packaged defaults. `experiment=`, `--config`, and `--resolved-config` are exclusive roots; exact replay preserves its resolved tree and rejects overrides.
- `config/resolved.yaml` is the immutable worker source of truth. Mutating commands persist the common config bundle only after collision checks; read-only inspection never writes.
- Completed checkpoints are explicit command inputs. Canonical training artifacts live in `checkpoints/`, `metrics/`, and `figures/ultralytics/`; framework-native output stays in `backend/ultralytics/`.
- `apply_inference_contract` restores only the model-coupled subset (`checkpoints.PINNED_SLICING_FIELDS`: `nominal_slice_size`, `image_format`); a divergence there raises `PreprocessingOverrideRefused`. `checkpoints.GEOMETRY_SLICING_FIELDS` (`level_policy`, `overlap_ratio`, `max_levels`) stay at their resolved values so infer/eval can re-slice an existing checkpoint. `export` passes `pin_geometry=True` and warns that a geometry override is inert. No slicing override may be silently discarded.
- Two preprocessing identities: `preprocessing_hash` is the trained pyramid, carried in the contract; `effective_preprocessing_hash` is the same source and version applied to the runtime slicing. They are equal unless geometry was overridden, and both ride every results row and the sidecar.
- Contract verification hashes a `SlicingConfig`/`ModelConfig` rebuilt from the contract, never the just-assigned `cfg` — self-comparison is a tautology that cannot fail.
- PyArrow schemas are always explicit and hand-written, built column-by-column; Arrow type inference is forbidden.
- Ultralytics is used stock (no dataset/trainer subclassing) and imported only behind the `Backend` protocol; core modules must import without the `[ultralytics]` extra installed.
- Coordinates: half-open `xyxy`, top-left origin, everywhere.

## Work Guidance

- Follow DESIGN-DOC-v1 section order; remaining open items are its `> [!NOTE]` callouts — route non-blocking ones to `QUESTIONS-FOR-SIDNEY.md`.
- Where SATI and dojo (`~/Projects/image-classifier-dojo/src/dojo`) share concepts (config compose/resolve, schema packages), mirror dojo's structure — Sidney's standing preference. Implemented so far: `config_schemas/` package (`root.py` + re-exporting `__init__`), `config_loader/` package (`compositor.py` / `resolver.py` / `conductor.py`).

## Verification

- `nix develop -c .venv/bin/python -m pytest tests/`.
- `SATI_RUN_TRIPWIRE=1 nix develop -c .venv/bin/python -m pytest tests/test_backends_train.py::test_ultralytics_coordinate_tripwire tests/test_checkpoints_lifecycle.py::test_real_ultralytics_finalized_checkpoint_round_trip`.
- `nix develop -c ruff check src tests`.

## Child DOX Index

- `config_loader/AGENTS.md` — compose → validate → resolve internals and searchpath/shadowing contract
