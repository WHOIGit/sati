"""Slice/label emission, pyramid boundary, data.yaml, end-to-end prepare (v1 §4.8)."""

import json

import cv2
import numpy as np
import pytest
import yaml

from sati.config_loader import compose_and_resolve
from sati.config_schemas import SlicingConfig, StartThenGeometricLevelPolicyConfig
from sati.hashing import preprocessing_hash
from sati.io import IngestError, ingest_coco
from sati.manifests import (
    SLICE_ANNOTATIONS_SCHEMA,
    SLICES_SCHEMA,
    read_manifest,
    read_metadata_sidecar,
    verify_metadata_hashes,
)
from sati.materialize import (
    build_level_pyramid,
    emit_slice,
    materialize_dataset,
    write_data_yaml,
)
from sati.tiling import build_level_stack, plan_frame


def _cfg(tmp_path, nominal=128, **extra):
    overrides = [
        "data.source_path=anns.json",
        "model.num_classes=2",
        f"slicing.nominal_slice_size={nominal}",
        "slicing.size_gate_px=4",
        *extra.pop("overrides", []),
    ]
    return compose_and_resolve(overrides, write=False, base_dir=tmp_path)


def _gradient(width, height):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.repeat(x[None, :], height, axis=0)
    return np.stack([img, img[::-1], np.full_like(img, 128)], axis=-1)


def _write_dataset(tmp_path, frames=((300, 200, "f1"), (100, 80, "f2"))):
    images, annotations = [], []
    for i, (w, h, _name) in enumerate(frames, start=1):
        cv2.imwrite(str(tmp_path / f"img{i}.png"), cv2.cvtColor(_gradient(w, h), cv2.COLOR_RGB2BGR))
        images.append({"id": i, "file_name": f"img{i}.png", "width": w, "height": h})
        annotations.append(
            {
                "id": 100 + i,
                "image_id": i,
                "category_id": 1,
                "bbox": [10.0, 10.0, 40.0, 30.0],
            }
        )
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "ring"}, {"id": 2, "name": "spiral"}],
            }
        )
    )
    return ingest_coco(tmp_path / "anns.json")


def test_pyramid_dims_and_level0_identity(tmp_path):
    cfg = _cfg(tmp_path)
    frame = _gradient(300, 200)
    levels = build_level_stack((300, 200), cfg.slicing)
    pyramid = build_level_pyramid(frame, levels)
    assert pyramid.pixels[0] is frame  # level 0 is the frame itself, no copy
    assert pyramid.fullres is frame
    for level in levels:
        assert pyramid.pixels[level.index].shape == (level.height, level.width, 3)


def test_pyramid_below_fullres_stack_starts_at_index_1():
    frame = _gradient(300, 200)
    slicing = SlicingConfig(
        nominal_slice_size=128,
        level_policy=StartThenGeometricLevelPolicyConfig(
            start_scale=0.5,
            subsequent_downsample_factor=3.0,
        ),
    )
    levels = build_level_stack((300, 200), slicing)
    # index 0 is reserved for full-res; a half-scale first level is level 1
    assert levels[0].index == 1 and levels[0].wh == (150, 100)
    pyramid = build_level_pyramid(frame, levels)
    assert 0 not in pyramid.pixels
    assert pyramid.pixels[1] is not frame
    assert pyramid.pixels[1].shape == (100, 150, 3)
    assert pyramid.fullres is frame  # full-res still available for sharp-source


def test_emit_slice_stretch_shapes(tmp_path):
    cfg = _cfg(tmp_path)
    plan = plan_frame((300, 200), cfg.slicing)
    pyramid = build_level_pyramid(_gradient(300, 200), plan.levels)
    for spec in plan.slices:
        emitted = emit_slice(spec, pyramid)
        assert emitted.shape == (spec.emitted_wh[1], spec.emitted_wh[0], 3)


def test_stretched_slices_source_from_finest_sufficient_level():
    # 4096×4096, nominal 1024, factor 2 → levels 0..2; level 2 (1024) unstretched
    # whole-level. With factor 3: levels 4096, 1365, 455 → level 2 whole-level
    # stretched: its sharp source must be level 1 (1365 ≥ 1024 both axes).
    from sati.materialize import _stretch_source

    slicing = SlicingConfig(
        nominal_slice_size=1024,
        level_policy={"kind": "auto", "downsample_factor": 3.0},
    )
    plan = plan_frame((4096, 4096), slicing)
    frame = np.zeros((4096, 4096, 3), np.uint8)
    pyramid = build_level_pyramid(frame, plan.levels)
    whole = next(s for s in plan.slices if s.is_whole_level)
    assert whole.level == 2 and whole.stretched_axes == ("x", "y")
    assert _stretch_source(whole, pyramid) == 1  # finest sufficient, not fullres

    # a frame whose finer levels are still < nominal on one axis → fullres fallback
    slicing_small = SlicingConfig(nominal_slice_size=1024)
    plan_small = plan_frame((1500, 900), slicing_small)
    pyr_small = build_level_pyramid(np.zeros((900, 1500, 3), np.uint8), plan_small.levels)
    stretched = next(s for s in plan_small.slices if s.level == 1)
    assert stretched.stretched_axes == ("x", "y")
    assert _stretch_source(stretched, pyr_small) is None  # → full-res frame


def test_sharp_source_reduces_whole_level_blur():
    # High-frequency checkerboard: emitting the whole-level slice from a finer
    # level must preserve more contrast than upscaling the coarse level itself.
    rng = np.random.default_rng(7)
    frame = (rng.integers(0, 2, (4096, 4096, 1)) * 255).astype(np.uint8).repeat(3, axis=2)
    slicing = SlicingConfig(
        nominal_slice_size=1024,
        level_policy={"kind": "auto", "downsample_factor": 3.0},
    )
    plan = plan_frame((4096, 4096), slicing)
    pyramid = build_level_pyramid(frame, plan.levels)
    whole = next(s for s in plan.slices if s.is_whole_level)
    sharp = emit_slice(whole, pyramid)
    # the old blurry path: upscale the coarse level's own pixels
    own = pyramid.pixels[whole.level]
    blurry = cv2.resize(own, whole.emitted_wh, interpolation=cv2.INTER_LINEAR)
    assert float(sharp.std()) > float(blurry.std())


def test_materialize_dataset_end_to_end(tmp_path):
    ds = _write_dataset(tmp_path)
    cfg = _cfg(tmp_path)
    root = materialize_dataset(ds, cfg, tmp_path / "out")

    slices = read_manifest(root / "manifests" / "slices.parquet", SLICES_SCHEMA)
    slice_anns = read_manifest(
        root / "manifests" / "slice_annotations.parquet", SLICE_ANNOTATIONS_SCHEMA
    )
    assert slices.num_rows > 0 and slice_anns.num_rows > 0

    # every slice image + label file exists; naming matches the manifest
    for row in slices.to_pylist():
        image_path = root / row["slice_uri"]
        assert image_path.is_file()
        assert image_path.name == f"{row['fullframe_id']}_L{row['level']}_S{row['slice_id']}.png"
        label = (
            root / "yolo" / "labels" / row["dataset_split"] / f"{image_path.stem}.txt"
        )
        assert label.is_file()
        for line in label.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5 and int(parts[0]) in (0, 1)
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])

    # emitted images are exactly nominal×nominal
    sample = cv2.imread(str(root / slices.to_pylist()[0]["slice_uri"]))
    assert sample.shape == (128, 128, 3)

    # sidecar hashes re-verify
    meta = read_metadata_sidecar(root / "manifests")
    assert set(meta.hashes) == {"class_map_hash", "dataset_hash", "preprocessing_hash"}
    assert all(verify_metadata_hashes(meta).values())
    assert list(meta.class_map.items()) == [("0", "ring"), ("1", "spiral")]

    # data.yaml embeds the dataset-level preprocessing hash
    spec = yaml.safe_load((root / "yolo" / "data.yaml").read_text())
    assert spec["sati_preprocessing_hash"] == preprocessing_hash(cfg.slicing)
    assert spec["names"] == {0: "ring", 1: "spiral"}
    assert spec["train"] == "images/training"


def test_zero_target_slice_gets_empty_label(tmp_path):
    ds = _write_dataset(tmp_path, frames=((300, 200, "f1"),))
    # single tiny annotation at top-left: far slices have zero targets
    cfg = _cfg(tmp_path)
    root = materialize_dataset(ds, cfg, tmp_path / "out")
    labels = list((root / "yolo" / "labels" / "training").glob("*.txt"))
    empties = [p for p in labels if p.read_text() == ""]
    assert empties  # negatives kept, written as empty files


def test_holdout_excluded_from_data_yaml_but_materialized(tmp_path):
    ds = _write_dataset(tmp_path)
    from sati.io import apply_split_override

    frames = apply_split_override(
        ds.frames, {"1": "training", "2": "holdout"}
    )
    ds = type(ds)(frames=frames, annotations=ds.annotations, class_map=ds.class_map)
    cfg = _cfg(tmp_path)
    root = materialize_dataset(ds, cfg, tmp_path / "out")
    assert list((root / "yolo" / "images" / "holdout").glob("*.png"))  # slices exist
    spec = yaml.safe_load((root / "yolo" / "data.yaml").read_text())
    # only train/val referenced; no holdout (or test) split key
    assert spec["train"] == "images/training" and spec["val"] == "images/validation"
    assert not {"test", "holdout"} & set(spec)


def test_dimension_mismatch_is_error(tmp_path):
    ds = _write_dataset(tmp_path, frames=((300, 200, "f1"),))
    # corrupt the manifest dims
    from dataclasses import replace

    frames = (replace(ds.frames[0], width=999),)
    ds = type(ds)(frames=frames, annotations=ds.annotations, class_map=ds.class_map)
    with pytest.raises(IngestError, match="manifest says 999x200"):
        materialize_dataset(ds, _cfg(tmp_path), tmp_path / "out")


def test_label_boxes_land_inside_slices(tmp_path):
    ds = _write_dataset(tmp_path)
    cfg = _cfg(tmp_path)
    root = materialize_dataset(ds, cfg, tmp_path / "out")
    slice_anns = read_manifest(
        root / "manifests" / "slice_annotations.parquet", SLICE_ANNOTATIONS_SCHEMA
    )
    for row in slice_anns.to_pylist():
        x1, y1, x2, y2 = row["bbox_slice_xyxy"]
        assert -1e-6 <= x1 <= x2 <= 128 + 1e-6
        assert -1e-6 <= y1 <= y2 <= 128 + 1e-6


def test_write_data_yaml_standalone(tmp_path):
    cfg = _cfg(tmp_path)
    path = write_data_yaml(tmp_path / "d", {0: "a"}, cfg.slicing)
    spec = yaml.safe_load(path.read_text())
    assert spec["path"].endswith("yolo")
