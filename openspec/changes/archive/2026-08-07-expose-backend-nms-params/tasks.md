## 1. Config surface

- [x] 1.1 Add `PredictionConfig` to `config_schemas/root.py`: `iou: float = 0.7`
  (`ge=0.0, le=1.0`) and `agnostic_nms: bool = False` — Ultralytics' current defaults,
  so this change alone moves no detection (D4).
- [x] 1.2 Add `prediction: PredictionConfig` to `RootConfig`.
- [x] 1.3 Add a `config_defaults/prediction/default.yaml` group with both fields
  written explicitly, so the values are visible in `composed.yaml` rather than
  materializing only from schema defaults.
- [x] 1.4 Do **not** add `prediction` to any `_ACTIVE_BLOCKS` entry
  (`hashing.py:140`). Add a comment at that map naming the omission as intentional,
  with the reason and a pointer to this change (D2).

## 2. Backend

- [x] 2.1 Widen the `Backend` protocol's `predict` signature (`backends.py:59`) with
  `iou: float` and `agnostic_nms: bool` as keyword parameters — scalars, not a config
  object, matching how `imgsz` and `confidence_threshold` are already passed (D5).
- [x] 2.2 Pass both through to `self._model.predict` in `UltralyticsBackend.predict`
  (`backends.py:150`), which currently supplies only `imgsz`, `conf`, and `verbose`.
- [x] 2.3 Update the module docstring, which describes what `predict` receives.

## 3. Callers

- [x] 3.1 Thread `cfg.prediction.iou` and `cfg.prediction.agnostic_nms` from
  `infer.py` into every `predict` call.
- [x] 3.2 No change needed: `evaluation.py` reaches the backend through
  `infer.infer_frame` (`evaluation.py:226`), so 3.1's threading covers eval too.
  Verified rather than assumed.
- [x] 3.3 Confirm `prediction` is included in the `infer`/`eval` resolved-config
  artifacts (`resolved.yaml`, `composed.yaml`, `resolved.json`) — this is the
  substantive win of the change and it must be verified, not assumed (D3).

## 4. Tests

- [x] 4.1 Guard test: `prediction` appears in no `_ACTIVE_BLOCKS` entry. This is the
  tripwire that makes a future decision to hash it deliberate rather than accidental
  (D2).
- [x] 4.2 Test that changing `prediction.iou` leaves `config_hash` and every
  compatibility identity **equal** — the test that documents the gap rather than
  hiding it. Reference D3's table in a comment so the test reads as intentional.
- [x] 4.3 Test that the values reach the backend: assert `model.predict` is called
  with the configured `iou` and `agnostic_nms`, using the existing backend test
  double.
- [x] 4.4 Test that a run with default `prediction` values produces detections
  identical to the pre-change code path, confirming D4's no-op claim.
- [x] 4.5 Test that `prediction` is present in the written `resolved.yaml`.

## 5. Documentation

- [x] 5.1 `DESIGN-DOC-v1.md` §7: the backend is no longer called with library-default
  NMS; state which parameters SATI now pins and why (drift protection, D4).
- [x] 5.2 Add `prediction.*` to the §3 config table.
- [x] 5.3 Add an explicit note to §4.7's hash-family section that `prediction` is a
  behavioral block deliberately outside the hash family, with the consequence spelled
  out: equal identities do not guarantee equal detections. The hash family's
  documentation is where a reader will look, so the caveat has to live there and not
  only in this change.

## 6. Verification and closeout

- [x] 6.1 Run `pytest tests/` and `ruff check src tests` in the Nix dev shell.
- [x] 6.2 Re-run inference on
  `datasets/infer-me/ODP-1209C-DP36-16-11H-3W-130-132cm_picked_N1of2_z200x_2d.tif`
  with default `prediction` values and confirm the output matches
  `runs/v1/combo1/default/2026-08-06_22-58-09_normal-galago_infer/` exactly —
  694 `level_merge` and 664 `final_merge` rows with identical boxes.
- [x] 6.3 Measure open question 1: run the same frame with `agnostic_nms: true` and
  record how many of the 347 cross-class duplicate pairs disappear at the slice rather
  than at recombination. Record the number in `design.md`; do not change the default
  in this change.
- [x] 6.4 Run `openspec validate --all --strict`, reconcile the main specs with these
  deltas, and leave the change ready for archive.
