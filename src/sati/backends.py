"""``Backend`` protocol + the stock-ultralytics implementation (v1 §7).

Ultralytics is used **stock**: no `DetectionTrainer` subclass, no injected
dataset, no loss changes. It is imported lazily so every other SATI module
works without the ``[ultralytics]`` extra installed.

``predict`` receives nominal×nominal RGB uint8 arrays. Slice size equals the
model ``imgsz``, so ultralytics applies no letterbox and returned boxes are
already in emitted-slice pixels (the tripwire test guards this contract).

Augmentation (settled 2026-07-03, §8 NOTE): v1 passes ``training.*`` through
untouched — stock ultralytics defaults apply, mosaic included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from sati.config_schemas import RootConfig


@dataclass(frozen=True)
class SliceDetection:
    """One raw detection in emitted-slice pixels, as returned by ``predict``."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


class Backend(Protocol):
    def train(self, data_yaml: Path, cfg: RootConfig) -> Path:
        """Train over the materialized YOLO tree; return the best checkpoint path."""
        ...

    def load(self, model_ref: Path | str) -> None:
        """Load weights for prediction."""
        ...

    def predict(
        self, slices: list[np.ndarray], *, imgsz: int, confidence_threshold: float
    ) -> list[list[SliceDetection]]:
        """Per input slice, detections in emitted-slice pixels."""
        ...

    def export(self, checkpoint: Path | str, *, fmt: str, imgsz: int) -> Path:
        """Export a checkpoint to a deployment format; return the artifact path."""
        ...


class UltralyticsBackend:
    """The one v1 backend: ultralytics, stock."""

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def _yolo(ref: str):
        try:
            from ultralytics import YOLO
        except ImportError as err:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "ultralytics is not installed; install the extra: pip install 'sati[ultralytics]'"
            ) from err
        return YOLO(ref)

    @staticmethod
    def _train_model_ref(cfg: RootConfig) -> str:
        # explicit weights win; else the arch's hub-pretrained checkpoint
        return cfg.model.weights or f"{cfg.model.arch}.pt"

    def train(self, data_yaml: Path, cfg: RootConfig) -> Path:
        model = self._yolo(self._train_model_ref(cfg))
        kwargs: dict = {
            "data": str(data_yaml),
            "imgsz": cfg.model.imgsz,
            "epochs": cfg.training.epochs,
            "batch": cfg.training.batch,
            "seed": cfg.training.seed,
            "project": str(cfg.output.run_dir),
            "name": "train",
            "exist_ok": True,
        }
        if cfg.training.lr is not None:
            kwargs["lr0"] = cfg.training.lr
        if cfg.training.device is not None:
            kwargs["device"] = cfg.training.device
        model.train(**kwargs)
        checkpoint = cfg.output.run_dir / "train" / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"training finished but checkpoint not found: {checkpoint}")
        return checkpoint

    def load(self, model_ref: Path | str) -> None:
        self._model = self._yolo(str(model_ref))

    def export(self, checkpoint: Path | str, *, fmt: str, imgsz: int) -> Path:
        # v1's all-square-slices rule pays off here: one fixed input shape, so
        # ONNX/TensorRT static shapes just work (§10)
        model = self._yolo(str(checkpoint))
        return Path(model.export(format=fmt, imgsz=imgsz))

    def predict(
        self, slices: list[np.ndarray], *, imgsz: int, confidence_threshold: float
    ) -> list[list[SliceDetection]]:
        if self._model is None:
            raise RuntimeError("call load() before predict()")
        # ultralytics treats raw ndarrays as BGR (cv2 convention); SATI is RGB
        bgr = [cv2.cvtColor(s, cv2.COLOR_RGB2BGR) for s in slices]
        results = self._model.predict(
            bgr, imgsz=imgsz, conf=confidence_threshold, verbose=False
        )
        out: list[list[SliceDetection]] = []
        for result in results:
            boxes = result.boxes
            detections = [
                SliceDetection(
                    bbox_xyxy=tuple(float(v) for v in xyxy),
                    score=float(conf),
                    class_id=int(cls),
                )
                for xyxy, conf, cls in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
                )
            ]
            out.append(detections)
        return out
