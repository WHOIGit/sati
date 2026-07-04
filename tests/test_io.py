"""COCO/YOLO ingestion adapters and split semantics (DESIGN-DOC-v1 §4, §4.6)."""

import json

import cv2
import numpy as np
import pytest

from sati.io import (
    IngestError,
    apply_split_override,
    detect_color_mode,
    ingest_coco,
    ingest_yolo,
    load_split_toml,
    read_image,
)


def _write_coco(path, image_ids, category_ids=(1, 3)):
    data = {
        "images": [
            {"id": i, "file_name": f"img{i}.png", "width": 640, "height": 480}
            for i in image_ids
        ],
        "annotations": [
            {
                "id": 100 + i,
                "image_id": i,
                "category_id": category_ids[0],
                "bbox": [10.0, 20.0, 30.0, 40.0],  # xywh
            }
            for i in image_ids
        ],
        "categories": [
            {"id": category_ids[0], "name": "ring"},
            {"id": category_ids[1], "name": "spiral"},
        ],
    }
    path.write_text(json.dumps(data))


def test_coco_single_file_xyxy_and_remap(tmp_path):
    _write_coco(tmp_path / "anns.json", [7, 9])
    ds = ingest_coco(tmp_path / "anns.json")
    assert [f.fullframe_id for f in ds.frames] == ["7", "9"]  # canonicalized to str
    assert all(f.dataset_split == "training" for f in ds.frames)
    # sparse COCO ids (1, 3) remap to contiguous 0-based, ordered by COCO id
    assert ds.class_map == {0: "ring", 1: "spiral"}
    ann = ds.annotations[0]
    assert ann.class_id == 0 and ann.class_name == "ring"
    assert ann.bbox_xyxy == (10.0, 20.0, 40.0, 60.0)  # xywh → xyxy
    assert ds.frames[0].source_uri.endswith("img7.png")


def test_coco_explicit_image_root(tmp_path):
    _write_coco(tmp_path / "anns.json", [1])
    images = tmp_path / "elsewhere"
    images.mkdir()
    ds = ingest_coco(tmp_path / "anns.json", image_root=images)
    assert ds.frames[0].source_uri == str((images / "img1.png").resolve())
    with pytest.raises(IngestError, match="image_root is not a directory"):
        ingest_coco(tmp_path / "anns.json", image_root=tmp_path / "nope")


def test_coco_per_split_directory(tmp_path):
    _write_coco(tmp_path / "training.json", [1])
    _write_coco(tmp_path / "validation.json", [2])
    ds = ingest_coco(tmp_path)
    splits = {f.fullframe_id: f.dataset_split for f in ds.frames}
    assert splits == {"1": "training", "2": "validation"}


def test_coco_duplicate_frame_across_splits_rejected(tmp_path):
    _write_coco(tmp_path / "training.json", [1])
    _write_coco(tmp_path / "validation.json", [1])
    with pytest.raises(IngestError, match="duplicate fullframe_id"):
        ingest_coco(tmp_path)


def test_coco_conflicting_category_names_rejected(tmp_path):
    _write_coco(tmp_path / "training.json", [1])
    data = json.loads((tmp_path / "training.json").read_text())
    data["categories"][0]["name"] = "not-ring"
    data["images"] = [{"id": 2, "file_name": "img2.png", "width": 10, "height": 10}]
    data["annotations"] = []
    (tmp_path / "validation.json").write_text(json.dumps(data))
    with pytest.raises(IngestError, match="category id 1"):
        ingest_coco(tmp_path)


def test_split_toml_and_override(tmp_path):
    _write_coco(tmp_path / "anns.json", [1, 2, 3])
    ds = ingest_coco(tmp_path / "anns.json")
    toml = tmp_path / "splits.toml"
    toml.write_text('training = ["1"]\nvalidation = ["2"]\n')
    split_map = load_split_toml(toml)
    with pytest.warns(UserWarning, match="'3' absent from split file"):
        frames = apply_split_override(ds.frames, split_map)
    assert {f.fullframe_id: f.dataset_split for f in frames} == {
        "1": "training",
        "2": "validation",
    }


def test_split_toml_accepts_image_filenames(tmp_path):
    _write_coco(tmp_path / "anns.json", [1, 2])
    ds = ingest_coco(tmp_path / "anns.json")
    toml = tmp_path / "splits.toml"
    toml.write_text('training = ["img1.png"]\nvalidation = ["img2.png"]\n')
    frames = apply_split_override(ds.frames, load_split_toml(toml))
    assert {f.fullframe_id: f.dataset_split for f in frames} == {
        "1": "training",
        "2": "validation",
    }


def test_split_toml_unknown_id_is_error(tmp_path):
    _write_coco(tmp_path / "anns.json", [1])
    ds = ingest_coco(tmp_path / "anns.json")
    toml = tmp_path / "splits.toml"
    toml.write_text('training = ["1", "999"]\n')
    with pytest.raises(IngestError, match="999"):
        apply_split_override(ds.frames, load_split_toml(toml))


def test_split_toml_duplicate_id_is_error(tmp_path):
    toml = tmp_path / "splits.toml"
    toml.write_text('training = ["1"]\nvalidation = ["1"]\n')
    with pytest.raises(IngestError, match="appears in both"):
        load_split_toml(toml)


def test_split_toml_unknown_key_is_error(tmp_path):
    toml = tmp_path / "splits.toml"
    toml.write_text('train = ["1"]\n')  # must be 'training'
    with pytest.raises(IngestError, match="unknown split keys"):
        load_split_toml(toml)


def _write_yolo_tree(root):
    for split, stem in (("train", "a"), ("val", "b")):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        cv2.imwrite(str(img_dir / f"{stem}.png"), np.zeros((100, 200, 3), np.uint8))
        (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ring\n  1: spiral\n"
    )


def test_yolo_ingest(tmp_path):
    _write_yolo_tree(tmp_path)
    ds = ingest_yolo(tmp_path / "data.yaml")
    assert ds.class_map == {0: "ring", 1: "spiral"}
    splits = {f.fullframe_id: f.dataset_split for f in ds.frames}
    assert splits == {"a": "training", "b": "validation"}
    frame_a = next(f for f in ds.frames if f.fullframe_id == "a")
    assert (frame_a.width, frame_a.height) == (200, 100)
    ann = next(a for a in ds.annotations if a.fullframe_id == "a")
    # cxcywh (0.5, 0.5, 0.2, 0.4) on 200×100 → xyxy (80, 30, 120, 70)
    assert ann.bbox_xyxy == pytest.approx((80.0, 30.0, 120.0, 70.0))
    assert ann.class_name == "ring"


def test_yolo_negative_frame(tmp_path):
    _write_yolo_tree(tmp_path)
    img_dir = tmp_path / "images" / "train"
    cv2.imwrite(str(img_dir / "neg.png"), np.zeros((50, 50, 3), np.uint8))
    ds = ingest_yolo(tmp_path)  # directory form
    assert "neg" in {f.fullframe_id for f in ds.frames}
    assert not [a for a in ds.annotations if a.fullframe_id == "neg"]


def test_read_image_and_color_mode(tmp_path):
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    cv2.imwrite(str(tmp_path / "img.png"), bgr)
    rgb = read_image(tmp_path / "img.png")
    assert rgb[0, 0].tolist() == [0, 0, 255]  # blue lands in the R-last slot
    assert detect_color_mode(rgb) == "RGB"
    assert detect_color_mode(np.full((5, 5, 3), 7, np.uint8)) == "grayscale"
    with pytest.raises(IngestError):
        read_image(tmp_path / "missing.png")
