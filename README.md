# SATI

Slice-Aware Training and Inference for object detection in very large images.

SATI tiles large input images into overlapping slices for training and
inference, then merges per-slice predictions back into full-image coordinates.
It targets YOLO-family models via `ultralytics` and is designed for scientific
imagery where objects are small relative to the full frame.

`src/sati/` holds the core library and CLI (`sati`).
`src/synth_dataset_builder/` holds the synthetic dataset pipeline (`sati-synth`).
Python dependencies are managed by `pyproject.toml`.

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
python -m pytest
```

Tests that require `ultralytics` or a GPU are skipped automatically when
those are not available.
