"""``Backend`` protocol + the stock-ultralytics implementation (v1 §7).

Ultralytics is used **stock**: no `DetectionTrainer` subclass, no injected
dataset, no loss changes. It is imported lazily so every other SATI module
works without the ``[ultralytics]`` extra installed.

``predict`` receives nominal×nominal RGB uint8 arrays. Slice size equals the
model ``imgsz``, so ultralytics applies no letterbox and returned boxes are
already in emitted-slice pixels (the tripwire test guards this contract).

Per-slice NMS (``iou``, ``agnostic_nms``) is supplied by SATI rather than left to
Ultralytics' defaults. Those defaults (``iou=0.7``, class-aware) previously ran
invisibly ahead of recombination, so a library change could move detections with
no config or hash difference. ``config.prediction`` now pins them — see
DESIGN-DOC-v1 §7.

Augmentation: omitted ``training.yolo_augment.*`` fields leave stock
Ultralytics defaults in force; explicit fields are passed through to
``YOLO.train`` for controlled ablations.
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


@dataclass(frozen=True)
class BackendTrainingArtifacts:
    root: Path
    best: Path
    last: Path


class Backend(Protocol):
    def train(
        self,
        data_yaml: Path,
        cfg: RootConfig,
        *,
        resume_from: Path | None = None,
    ) -> BackendTrainingArtifacts:
        """Train over the materialized YOLO tree; return native artifacts."""
        ...

    def load(self, model_ref: Path | str) -> None:
        """Load weights for prediction."""
        ...

    def predict(
        self,
        slices: list[np.ndarray],
        *,
        imgsz: int,
        confidence_threshold: float,
        iou: float,
        agnostic_nms: bool,
    ) -> list[list[SliceDetection]]:
        """Per input slice, detections in emitted-slice pixels.

        ``iou`` and ``agnostic_nms`` are the backend's own per-slice NMS
        parameters. They are part of the contract rather than left to the
        library's defaults, so an upstream default change cannot move detections
        silently.
        """
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
        weights = cfg.model.weights
        if weights.source == "none":
            return f"{cfg.model.architecture.name}.yaml"
        if weights.uri:
            return weights.uri
        return f"{cfg.model.architecture.name}.pt"

    def train(
        self,
        data_yaml: Path,
        cfg: RootConfig,
        *,
        resume_from: Path | None = None,
    ) -> BackendTrainingArtifacts:
        model = self._yolo(str(resume_from) if resume_from else self._train_model_ref(cfg))
        if cfg.training_outputs.backend.dir is None:
            raise RuntimeError("training_outputs.backend.dir was not resolved")
        kwargs: dict = {
            "data": str(data_yaml),
            "imgsz": cfg.model.imgsz,
            "epochs": cfg.training.epochs,
            "batch": cfg.training.batch,
            "seed": cfg.runtime.seed,
            "project": str(cfg.training_outputs.backend.dir),
            "name": "ultralytics",
            "exist_ok": bool(resume_from),
        }
        if resume_from is not None:
            kwargs["resume"] = str(resume_from)
        if cfg.training.lr is not None:
            kwargs["lr0"] = cfg.training.lr
        if cfg.training.device is not None:
            kwargs["device"] = cfg.training.device
        if cfg.training.patience is not None:
            kwargs["patience"] = cfg.training.patience
        kwargs.update(cfg.training.yolo_augment.explicit_train_kwargs())
        model.train(**kwargs)
        root = cfg.training_outputs.backend.dir / "ultralytics"
        best = root / "weights" / "best.pt"
        last = root / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(
                f"training finished but native checkpoints were not found under {root}"
            )
        return BackendTrainingArtifacts(root=root, best=best, last=last)

    def load(self, model_ref: Path | str) -> None:
        self._model = self._yolo(str(model_ref))

    def export(self, checkpoint: Path | str, *, fmt: str, imgsz: int) -> Path:
        # v1's all-square-slices rule pays off here: one fixed input shape, so
        # ONNX/TensorRT static shapes just work (§10)
        model = self._yolo(str(checkpoint))
        return Path(model.export(format=fmt, imgsz=imgsz))

    def predict(
        self,
        slices: list[np.ndarray],
        *,
        imgsz: int,
        confidence_threshold: float,
        iou: float,
        agnostic_nms: bool,
    ) -> list[list[SliceDetection]]:
        if self._model is None:
            raise RuntimeError("call load() before predict()")
        # ultralytics treats raw ndarrays as BGR (cv2 convention); SATI is RGB
        bgr = [cv2.cvtColor(s, cv2.COLOR_RGB2BGR) for s in slices]
        results = self._model.predict(
            bgr,
            imgsz=imgsz,
            conf=confidence_threshold,
            iou=iou,
            agnostic_nms=agnostic_nms,
            verbose=False,
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
