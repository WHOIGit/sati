## Why

`UltralyticsBackend.predict` (`backends.py:150`) calls

```python
self._model.predict(bgr, imgsz=imgsz, conf=confidence_threshold, verbose=False)
```

with **no `iou` and no `agnostic_nms`**. Ultralytics therefore runs its own
per-slice NMS at library defaults — `iou=0.7`, class-aware — and those two values
appear in no SATI config, no resolved-config artifact, and no hash. Three
consequences:

1. **A behavioral input is invisible.** Nothing in a run directory records what
   per-slice NMS did. `resolved.yaml` is meant to be the complete rendered truth
   passed to the worker; for this it is silently incomplete.
2. **A version bump can change results with every digest unchanged.** These are
   library defaults, not SATI's. If Ultralytics changes `iou=0.7`, detections move
   and `config_hash`, `model_config_hash`, and `preprocessing_hash` all stay equal.
   The `DESIGN-DOC-v1.md` §7 API-drift tripwire guards the coordinate contract, not
   these values.
3. **It is upstream of a problem being solved downstream.** The 347 cross-class
   duplicate pairs measured in `recombination-match-and-merge` are produced *by*
   class-aware per-slice NMS. That change fixes them at recombination time, which is
   the right place — but the parameter that generates them should still be visible
   and adjustable, so the two can be reasoned about separately.

This change makes both values explicit configuration. It deliberately does **not**
put them in any hash yet — see Non-Goals.

## What Changes

- **NEW config** A `prediction` block on the root config with `iou: float = 0.7` and
  `agnostic_nms: bool = False`. Defaults reproduce Ultralytics' current behavior
  exactly, so adopting this change alone moves no detection.
- **NEW** `Backend.predict` gains `iou` and `agnostic_nms` parameters, passed
  through to `model.predict`. The protocol signature widens; there is one
  implementation.
- **NOT HASHED, deliberately.** `prediction` is **not** added to any command's entry
  in `_ACTIVE_BLOCKS` (`hashing.py:140`), so it is excluded from `config_hash` by
  omission from that allowlist — no new marker and no hash-machinery change. A
  comment at the tuple and a guard test record that the omission is intentional, so
  a future reader cannot "fix the oversight" and silently move every digest.
- **RECORDED even though unhashed.** Because the block is part of the root config,
  its values land in `<output>/config/resolved.yaml`, `composed.yaml`, and
  `resolved.json` like any other field. This is the substantive win: the values go
  from *invisible* to *recorded per run*, independent of the hashing question.
- Docs: `DESIGN-DOC-v1.md` §7 (the backend is no longer called with library-default
  NMS), the §3 config table, and an explicit note in §4.7 that `prediction` is a
  behavioral block outside the hash family, with the reason.

## Capabilities

### New Capabilities
<!-- None: this exposes existing backend behavior as configuration. -->

### Modified Capabilities

- `model-training`: the backend protocol's `predict` contract includes the
  per-slice NMS parameters rather than deferring them to library defaults.
- `config-and-cli`: the root schema gains a `prediction` block.
- `results-provenance`: the hash family is explicitly documented as **not** covering
  `prediction`, so a reader knows two runs with equal identities may differ in
  per-slice NMS. Stating the gap is what makes it safe to defer.

## Non-Goals

- **Hashing `prediction`.** Deferred by decision. The honest consequence, recorded
  rather than glossed: two runs differing only in `prediction.iou` will carry
  identical `config_hash` values and produce different detections, so
  content-addressed provenance is incomplete for exactly these two fields. This is
  an improvement on today only because the values become *recorded*; it does not
  close the reproducibility gap. Revisiting is cheap — adding `prediction` to the
  `infer`/`eval`/`train` tuples and bumping `CONFIG_HASH` is the whole change.
- Folding them into `preprocessing_hash`. That hash covers the pixel path — slicing,
  levels, stretch, projection. Per-slice NMS is a post-inference decision on model
  outputs and does not belong there; a `prediction_hash` scoped like
  `recombination_hash` would be the right home if and when this is hashed.
- Exposing any other Ultralytics predict argument (`max_det`, `half`, `augment`,
  TTA). Absent stays unsupported.
- Changing any default. `iou=0.7` and `agnostic_nms=False` match the library today.
- Retuning `iou`. Whether 0.7 is right for dense small objects in gigapixel imagery
  is a real question this change only makes *askable*.

## Impact

- `src/sati/backends.py`: the `Backend` protocol's `predict` signature and the
  Ultralytics implementation (`backends.py:143`, `:150`).
- `src/sati/config_schemas/root.py`: `PredictionConfig`, plus `prediction` on
  `RootConfig`.
- `src/sati/config_defaults/`: a `prediction/` group with a `default.yaml`.
- `src/sati/infer.py` and `evaluation.py`: pass the config values into `predict`.
- `src/sati/hashing.py`: a comment at `_ACTIVE_BLOCKS` recording the intentional
  omission; no functional change.
- `tests/`: a guard test asserting `prediction` is absent from every
  `_ACTIVE_BLOCKS` entry, and a test that changing `prediction.iou` leaves every
  hash equal while changing the detections — the test that documents the gap
  instead of hiding it.
- No new dependency. No existing hash, schema version, or artifact changes, so
  nothing to migrate.
- Independent of `recombination-match-and-merge` and `infer-render-artifacts`; the
  three can land in any order.
