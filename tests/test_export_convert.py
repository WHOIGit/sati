"""Portable export and no-slicing conversion contracts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from sati.checkpoints import build_inference_contract
from sati.config_loader import compose_and_resolve
from sati.convert import execute_convert
from sati.export import execute_export
from sati.io import IngestError


def _cfg(tmp_path, *, command="export", overrides=()):
    output_override = (
        f"export_outputs.dir={tmp_path}/export"
        if command == "export"
        else f"prepare_outputs.dir={tmp_path}/convert"
    )
    return compose_and_resolve(
        [
            f"data.source.uri={tmp_path}/anns.json",
            "model.num_classes=2",
            output_override,
            *overrides,
        ],
        command=command,
        base_dir=tmp_path,
    )


class ExportBackend:
    def export(self, checkpoint, *, fmt, imgsz):
        artifact = Path(checkpoint).with_suffix(f".{fmt}")
        artifact.write_bytes(b"exported-" + fmt.encode())
        return artifact


def test_execute_export_and_portable_sidecar(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _cfg(tmp_path)
    contract = build_inference_contract(cfg, class_mapping={0: "ring", 1: "spiral"})
    artifact, sidecar = execute_export(
        cfg,
        checkpoint=checkpoint,
        fmt="onnx",
        backend=ExportBackend(),
        contract=contract,
    )
    assert artifact.parent == cfg.export_outputs.models.dir
    spec = yaml.safe_load(sidecar.read_text())
    assert spec["class_mapping"] == {"0": "ring", "1": "spiral"}
    assert spec["export"]["format"] == "onnx"
    assert spec["export"]["checkpoint_hash"].startswith("sha256:")
    assert spec["export"]["model_hash"].startswith("sha256:")


def _tiny_coco(tmp_path):
    cv2.imwrite(str(tmp_path / "image.png"), np.zeros((50, 80, 3), np.uint8))
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "image.png", "width": 80, "height": 50}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 15],
                    }
                ],
                "categories": [{"id": 1, "name": "ring"}],
            }
        )
    )


def test_convert_coco_manifests_and_yolo(tmp_path):
    _tiny_coco(tmp_path)
    to_manifests = _cfg(tmp_path, command="convert")
    manifests = execute_convert(to_manifests, to="manifests")
    assert (manifests / "fullframes.parquet").is_file()
    assert not (to_manifests.prepare_outputs.dir / "yolo" / "images").exists()

    to_yolo = _cfg(
        tmp_path,
        command="convert",
        overrides=[
            f"data.source.uri={to_manifests.prepare_outputs.dir}",
            f"prepare_outputs.dir={tmp_path}/yolo-convert",
        ],
    )
    yolo = execute_convert(to_yolo, to="yolo")
    spec = yaml.safe_load((yolo / "data.yaml").read_text())
    assert spec["names"] == {0: "ring"}
    assert list((yolo / "labels" / "training").glob("*.txt"))


def test_convert_unknown_target(tmp_path):
    _tiny_coco(tmp_path)
    with pytest.raises(IngestError, match="unknown convert target"):
        execute_convert(_cfg(tmp_path, command="convert"), to="bad")
