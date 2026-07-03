# Questions for Sidney

Non-blocking questions and provisional decisions, per standing instruction. Each item states
what was done in the meantime, so nothing here blocks work. Settled items keep their number,
struck through, with the resolution recorded. Entries from 2026-07-02 23:15 EDT onward carry
full timestamps (`YYYY-MM-DD HH:MM TZ`); earlier headings are date-only.

## 2026-07-02 — pyproject bootstrap

1. ~~`requires-python` floor~~ — **settled 2026-07-02: `>=3.12` OK** (dev on 3.14).
2. ~~opencv-python vs headless~~ — **settled 2026-07-02:** use whatever OpenCV ultralytics
   pulls; it kept `opencv-python 5.0.0`, so plain (non-headless) in core stands.
3. **OpenCV 5.0.0 API drift watch** — still open as a to-verify: check
   `fillPoly`/`INTER_AREA`/`resize` behavior when render code lands, flag any 4.x→5.x shifts.
4. ~~ultralytics install timing~~ — **settled 2026-07-02:** Sidney installed it
   (ultralytics 8.4.86, torch 2.12.1) into `.venv`.
5. ~~Project metadata~~ — **settled 2026-07-02:** author Sidney Batchelder, MIT license —
   now in `pyproject.toml`.

## 2026-07-02 — config_schemas + config_loader (dojo-structured)

6. ~~Rename `config_defaults/` to `configs/`?~~ — **settled 2026-07-02: keep
   `config_defaults/`.**
7. ~~Fragment shadowing needs the primary config~~ — **settled 2026-07-02: hybrid scheme.**
   `experiment=name` makes `experiment/name` the primary root (dojo-style, so local
   experiments compose packaged groups with no local `config.yaml`); selector-less runs
   keep the packaged `config.yaml` root. Group-fragment shadowing without an experiment
   still requires a local `config.yaml` copy — inherent Hydra constraint.
8. **`--resolved-config` re-runs get a fresh `run_id`** (old one is discarded; hashes are
   unaffected since run_id is derived-only and excluded). Alternative would be reusing the
   original run_id, which risks clobbering the original run's outputs — flag if you
   disagree.
9. ~~`model.weights` machine-specific hash hole~~ — **settled 2026-07-02 (Sidney's rule):**
   file-backed weights contribute their *byte* hash (`checkpoint_hash`) to
   `config_hash`/`model_config_hash` instead of the path string; hub asset names hash
   verbatim. Implemented in `sati/hashing.py` + tests; noted back to dojo via its root
   AGENTS.md "Communications from the SATI Agent" section.

## 2026-07-02 — sati.tiling

## 2026-07-03 14:37 EDT — sati.export / convert

22. **`convert --to yolo` symlinks images** instead of copying (fast, no duplication;
    linux-only assumption). Copying would be portable but heavy for 100MP frames.
23. **Export's `inference_config.yaml` class map** comes from a prepared root at
    `data.source_path` when one is there; otherwise `{}` (a bare checkpoint doesn't
    carry SATI's class map). If you want export to *require* a class map, say so.

## 2026-07-03 14:34 EDT — sati.evaluation

20. **Eval split is fixed to `validation`.** §3.3 defines no eval-split knob and absent =
    unsupported, so v1 always evaluates the validation split. Holdout scoring would need
    a schema field (e.g. `evaluation.split`) — say the word and it's a 5-line change.
21. **GT-recall size bins use a log2 long-side ladder** (16/32/64/…/1024+ px) rather than
    COCO's 3 area bins — finer resolution on SATI's scale axis. Purely diagnostic.

## 2026-07-03 09:38 EDT — sati.infer

18. **`level_merge` rows are written unconditionally.** §9 enumerates "final_merge
    (+ frame_metadata, + optional cached slice_prediction)" without naming level_merge,
    but §4.5 defines them as a first-class record_type and they're cheap (one row per
    Stage-1 winner). I write them always; if you'd rather gate them behind a config
    knob like slice_predictions, say so.
19. **Infer on new imagery labels rows `dataset_split="holdout"`** (no split exists for
    fresh imagery; holdout = "never scored" which matches). Alternative: a distinct
    `"inference"` split value — would require widening the split vocabulary.

## 2026-07-03 03:46 EDT — sati.backends / train / CLI

16. **`sati train` finds its dataset via `data.source_path` pointing at the prepared
    root** (the prepare run's `output.run_dir`, containing `yolo/data.yaml`) or directly
    at a `data.yaml`. Keeps train fully config-driven (no extra CLI arg); the chain is
    `prepare output.run_dir=X` → `train data.source_path=X`.
17. **Model init: explicit `model.weights` wins; else `f"{arch}.pt"`** (hub-pretrained,
    the standard ultralytics default) rather than `f"{arch}.yaml"` (from-scratch). Set
    `model.weights` to a `.yaml` to train from scratch.

## 2026-07-03 03:33 EDT — sati.materialize

14. **Stretch interpolation = `INTER_LINEAR`.** §6.1 says stretch uses `cv2.resize` but
    names no interpolation. Downsampling is spec'd INTER_AREA; for the (upscaling)
    stretch I chose INTER_LINEAR. INTER_CUBIC would be sharper/slower — say the word.
15. **`dataset_hash` source = emitted slice images** (relpath+size under `yolo/images/`).
    §4.7 says "manifest file identities" — I read that as "identities of the files the
    dataset comprises", and the emitted slices are what training/eval actually consume.
    Alternative reading: the four `manifests/*.parquet` files themselves. Current choice
    is stricter (any slicing change shifts the hash even before manifests are read).

## 2026-07-03 03:16 EDT — sati.io

12. ~~COCO image root~~ — **settled 2026-07-03 03:23 EDT (Sidney): added
    `data.image_root`** (optional `SatiPath`, COCO-only — a validator rejects it for
    `source_format=yolo`). Default remains each annotation file's own directory.
13. **YOLO `fullframe_id` = image filename stem**, required unique across all splits
    (hard error otherwise). COCO uses the numeric image id as a string.

## 2026-07-03 03:13 EDT — sati.manifests

11. **`slicing.image_format` excluded from `preprocessing_hash`.** §4.7 lists "nominal
    slice, stack, overlap, stretch policy" — format isn't named, so I excluded it. But
    the hash gates *ensembling of raw slice predictions*, and png-vs-jpeg pixels differ
    slightly; if you consider codec part of preprocessing identity, say so and I'll add
    it (recipe bumps to `/2`).

10. **Size gate evaluated on the *clipped* (visible) box.** v1 §6.1 says "shorter side,
    evaluated at the level before stretch" without saying clipped or unclipped. I gate on
    the clipped visible box (a large object whose visible sliver is thin gets dropped by
    the gate and usually by `min_area_ratio` too), which matches SAHI's
    visible-box behavior. Flag if you want the unclipped level box instead.
