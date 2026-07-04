# src/sati

## Purpose

The SATI package: slice-aware prepare/convert/train/eval/infer/export plus a read-only `inspect` sub-app, per `DESIGN-DOC-v1.md` (v2 delta comes later).

## Ownership

Owns all `sati.*` modules. Target layout per DESIGN-DOC-v1 §12: `config_schemas/`, `config_defaults/`, `config_loader/`, `io`, `manifests`, `hashing`, `tiling` (incl. recombination), `materialize`, `backends`, `train`/`infer`/`evaluation`, `inspect`, `cli`. Implemented so far: `config_schemas/`, `config_loader/`, `config_defaults/`, `hashing` (full §4.7 family), `tiling` (geometry + two-stage NMS recombination), `manifests`, `io`, `materialize` (pixel seam + `execute_prepare`), `backends` (protocol + stock ultralytics, coordinate tripwire), `train` (preprocessing gate + `execute_train`), `infer` (sliced inference engine + results rows/sidecar), `evaluation` (pycocotools mAP + §11 diagnostics), `export` (artifact + inference_config.yaml), `convert` (COCO/YOLO ↔ manifests), `inspect` (§5.1 read-only sub-app), `cli` (full §5 surface: `prepare convert train eval infer export inspect`). **All v1 §12 modules implemented.**

## Local Contracts

- Pydantic v2 schemas are the single source of truth: `extra="forbid"`, absent = unsupported. Validation once at the boundary; downstream code takes typed models, never `DictConfig`.
- All path-typed schema fields use the `SatiPath` annotated alias — bare `pathlib.Path` in a schema is a bug (config_hash strips by annotation; DESIGN-DOC-v1 §4.7).
- Slicing supports `level_policy.kind=auto` and `start_then_geometric`; **each policy owns its own factor** (`level_policy.downsample_factor` / `.subsequent_downsample_factor` — no shared top-level knob). Level index 0 ⇔ full-res always; below-full-res stacks number from 1, so pyramids are keyed by level index (`LevelPyramid`), never list position.
- Stretched slice windows are emitted via the sharp-source rule (finest sufficient level, else full-res; §6.1) — never upscaled from their own level's pixels.
- Split TOML entries may name source image filenames, source image stems, or SATI fullframe ids; aliases must resolve unambiguously.
- `training.yolo_augment.*` fields are optional pass-throughs to stock Ultralytics train kwargs; omitted fields preserve Ultralytics defaults.
- Packaged Hydra defaults resolve via `importlib.resources.files("sati")` to a real directory; hard error on zip imports (§3.2). Compose API only, never `@hydra.main`.
- PyArrow schemas are always explicit and hand-written, built column-by-column; Arrow type inference is forbidden.
- Ultralytics is used stock (no dataset/trainer subclassing) and imported only behind the `Backend` protocol; core modules must import without the `[ultralytics]` extra installed.
- Coordinates: half-open `xyxy`, top-left origin, everywhere.

## Work Guidance

- Follow DESIGN-DOC-v1 section order; remaining open items are its `> [!NOTE]` callouts — route non-blocking ones to `QUESTIONS-FOR-SIDNEY.md`.
- Where SATI and dojo (`~/Projects/image-classifier-dojo/src/dojo`) share concepts (config compose/resolve, schema packages), mirror dojo's structure — Sidney's standing preference. Implemented so far: `config_schemas/` package (`root.py` + re-exporting `__init__`), `config_loader/` package (`compositor.py` / `resolver.py` / `conductor.py`).

## Verification

- `.venv/bin/python -m pytest tests/` (116 tests + 1 gated: config, hashing, tiling, recombination, manifests, io, materialize, backends/train/CLI, infer, evaluation incl. the oracle end-to-end round trip, export/convert, inspect).
- `SATI_RUN_TRIPWIRE=1 .venv/bin/python -m pytest tests/test_backends_train.py -k tripwire` — ultralytics coordinate-contract tripwire (downloads hub weights; run after any ultralytics version bump).
- `.venv/bin/ruff check src tests` must be clean.

## Child DOX Index

- `config_loader/AGENTS.md` — compose → validate → resolve internals and searchpath/shadowing contract
