# src/sati

## Purpose

The SATI package: slice-aware prepare/convert/train/eval/infer/export plus a read-only `inspect` sub-app, per `DESIGN-DOC-v1.md` (v2 delta comes later).

## Ownership

Owns all `sati.*` modules. Target layout per DESIGN-DOC-v1 §11: `config_schemas`, `config_defaults/`, `config_loader`, `manifests`, `tiling`, `recombine`, `backends`, `metrics`, `inspect`, `cli`.

## Local Contracts

- Pydantic v2 schemas are the single source of truth: `extra="forbid"`, absent = unsupported. Validation once at the boundary; downstream code takes typed models, never `DictConfig`.
- All path-typed schema fields use the `SatiPath` annotated alias — bare `pathlib.Path` in a schema is a bug (config_hash strips by annotation; DESIGN-DOC-v1 §4.7).
- Packaged Hydra defaults resolve via `importlib.resources.files("sati")` to a real directory; hard error on zip imports (§3.2). Compose API only, never `@hydra.main`.
- PyArrow schemas are always explicit and hand-written, built column-by-column; Arrow type inference is forbidden.
- Ultralytics is used stock (no dataset/trainer subclassing) and imported only behind the `Backend` protocol; core modules must import without the `[ultralytics]` extra installed.
- Coordinates: half-open `xyxy`, top-left origin, everywhere.

## Work Guidance

- Follow DESIGN-DOC-v1 section order; remaining open items are its `> [!NOTE]` callouts — route non-blocking ones to `QUESTIONS-FOR-SIDNEY.md`.

## Verification

(none yet — update when the test suite exists)

## Child DOX Index

(none)
