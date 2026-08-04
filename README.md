# SATI

Slice-Aware Training and Inference for object detection in very large images.

SATI tiles large input images into overlapping slices for training and
inference, then merges per-slice predictions back into full-image coordinates.
It targets YOLO-family models via `ultralytics` and is designed for scientific
imagery where objects are small relative to the full frame.

`src/sati/` holds the core library and CLI (`sati`).
`src/synth_dataset_builder/` holds the synthetic dataset pipeline (`sati-synth`).
Python dependencies are managed by `pyproject.toml`.

## Command and configuration workflow

SATI uses Hydra composition with a dojo-style, config-first CLI. Project
experiments live under `configs/experiment/`; packaged starter defaults live
under `src/sati/config_defaults/`.

```bash
# Prepare the reusable dataset product named by an experiment.
sati prepare experiment=v1/manual/fullres_bbox

# Train it, with an ordinary Hydra inline override.
sati train experiment=v1/manual/fullres_bbox training.epochs=10

# Consume a completed checkpoint as command input, not model-init config.
sati eval experiment=v1/manual/fullres_bbox --checkpoint runs/.../checkpoints/best.abc123.pt
sati infer experiment=v1/manual/fullres_bbox \
  --checkpoint runs/.../checkpoints/best.abc123.pt \
  --input ./new-images
```

The three root sources are mutually exclusive:

- `experiment=name` selects `./configs/experiment/name.yaml` with packaged
  group fallback.
- `--config path.yaml` composes an authored root and permits inline overrides.
- `--resolved-config <run>/config/resolved.yaml` exactly replays an
  instantiated config, preserving command, IDs, hashes, and paths. It rejects
  behavioral overrides; branch from `config/composed.yaml` with `--config`.

Dash-free `key=value` tokens are Hydra overrides. The small set of
dash-prefixed options represents command I/O or lifecycle control:
`--checkpoint`, `--input`, `--output`, `--format`, `--resume`, `--clobber`,
and config-source options.

Every mutating invocation writes a collision-safe output envelope containing
an atomic lifecycle manifest and:

```text
config/
  composed.yaml
  resolved.yaml      # authoritative configuration actually given to the worker
  resolved.json      # the same tree in JSON
  cli.txt
  overrides.txt
```

Training additionally exposes stable `checkpoints/`, `metrics/metrics.csv`,
and optional `figures/ultralytics/`; native Ultralytics output is isolated
under `backend/ultralytics/`. Run `sati inspect --help` for read-only artifact
summaries.

---

## Installation

### Nix / NixOS

A `flake.nix` is provided. Enter the dev shell with:

```bash
nix develop
```

This pins Python 3.12, provides system libraries needed by opencv-python
(`libGL`, `glib`, `libxcb`, `libx11`, `libxext`), creates `.venv`, and
runs `pip install -e '.[dev,synth]'` automatically on first entry.

`ultralytics` (YOLO training and inference) requires PyTorch, which must be
installed separately because the correct CUDA wheel depends on your driver
version. After the initial `nix develop`, install torch outside the shell:

```bash
# Example for CUDA 12.8 (cu128) — see https://pytorch.org/get-started/locally/
# for the selector matching your driver.
.venv/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

Re-entering `nix develop` after torch is installed will detect it and
automatically run `pip install -e '.[ultralytics]'`.

### Manual (virtualenv)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,synth]"
```

Install PyTorch for your CUDA version, then the YOLO extra:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ultralytics]"
```

---

## Testing

```bash
nix develop -c .venv/bin/python -m pytest tests/
nix develop -c ruff check src tests
```

Tests that require `ultralytics` or a GPU are skipped automatically when
those are not available.
