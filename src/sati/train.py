"""Stock-Ultralytics training with canonical SATI artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import yaml

from sati.backends import Backend, BackendTrainingArtifacts, UltralyticsBackend
from sati.checkpoints import (
    build_inference_contract,
    finalize_checkpoint,
    write_checkpoint_metadata,
)
from sati.config_schemas import RootConfig
from sati.errors import SatiError
from sati.hashing import preprocessing_hash
from sati.manifests import read_metadata_sidecar


class TrainRefusedError(SatiError):
    """A compatibility or resume gate failed before loading a model."""


def locate_data_yaml(cfg: RootConfig) -> Path:
    root = cfg.data.prepared.uri
    if root is None:
        raise TrainRefusedError("data.prepared.uri was not resolved")
    candidate = root / "yolo" / "data.yaml"
    if candidate.is_file():
        return candidate
    raise TrainRefusedError(
        f"no prepared dataset at {root}: expected <product>/yolo/data.yaml "
        "(run `sati prepare` first)"
    )


def check_prepared_gate(data_yaml: Path, cfg: RootConfig) -> None:
    spec = yaml.safe_load(data_yaml.read_text())
    expected = {
        "dataset_hash": cfg.data.prepared.dataset_hash,
        "prepared_dataset_hash": cfg.data.prepared.prepared_dataset_hash,
        "preprocessing_hash": preprocessing_hash(cfg.slicing),
    }
    observed = {
        "dataset_hash": spec.get("sati_dataset_hash"),
        "prepared_dataset_hash": spec.get("sati_prepared_dataset_hash"),
        "preprocessing_hash": spec.get("sati_preprocessing_hash"),
    }
    missing = [name for name, value in observed.items() if value is None]
    if missing:
        raise TrainRefusedError(
            f"{data_yaml} is not a current SATI-prepared tree; missing {', '.join(missing)}"
        )
    mismatches = [
        f"{name}: expected {expected[name]}, observed {observed[name]}"
        for name in expected
        if expected[name] is not None and expected[name] != observed[name]
    ]
    if mismatches:
        raise TrainRefusedError("prepared product compatibility mismatch — " + "; ".join(mismatches))


def check_preprocessing_gate(data_yaml: Path, cfg: RootConfig) -> None:
    """Focused public gate retained under its descriptive v1 name."""
    check_prepared_gate(data_yaml, cfg)


_METRIC_NAMES = {
    "epoch": "epoch",
    "time": "time_seconds",
    "train/box_loss": "train.box_loss",
    "train/cls_loss": "train.class_loss",
    "train/dfl_loss": "train.dfl_loss",
    "metrics/precision(B)": "validation.precision",
    "metrics/recall(B)": "validation.recall",
    "metrics/mAP50(B)": "validation.map50",
    "metrics/mAP50-95(B)": "validation.map50_95",
    "val/box_loss": "validation.box_loss",
    "val/cls_loss": "validation.class_loss",
    "val/dfl_loss": "validation.dfl_loss",
}


def normalize_epoch_metrics(native_csv: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with native_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        native_fields = reader.fieldnames or []
    fields = [
        _METRIC_NAMES.get(field.strip(), f"ultralytics.{field.strip()}")
        for field in native_fields
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    _METRIC_NAMES.get(key.strip(), f"ultralytics.{key.strip()}"): value
                    for key, value in row.items()
                }
            )
    return destination


def copy_native_figures(native_root: Path, figures_root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    destination = figures_root / "ultralytics"
    copied: list[Path] = []
    for source in sorted(native_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in suffixes:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def canonical_last_checkpoint(cfg: RootConfig) -> Path:
    if cfg.training_outputs.checkpoints.dir is None:
        raise TrainRefusedError("training checkpoint directory is unresolved")
    metadata_path = cfg.training_outputs.checkpoints.dir / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        raise TrainRefusedError("resume requires canonical checkpoint_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    entry = metadata.get("checkpoints", {}).get("last")
    if not entry:
        raise TrainRefusedError("resume requires a canonical last checkpoint")
    checkpoint = cfg.training_outputs.checkpoints.dir / entry["path"]
    if not checkpoint.is_file():
        raise TrainRefusedError(f"resume checkpoint is missing: {checkpoint}")
    return checkpoint


def execute_train(
    cfg: RootConfig,
    backend: Backend | None = None,
    *,
    resume_from: Path | None = None,
) -> tuple[Path, str]:
    backend = backend or UltralyticsBackend()
    data_yaml = locate_data_yaml(cfg)
    check_prepared_gate(data_yaml, cfg)
    prepared_metadata = read_metadata_sidecar(cfg.data.prepared.uri / "manifests")
    class_mapping = {
        int(index): name for index, name in prepared_metadata.class_mapping.items()
    }
    native: BackendTrainingArtifacts = backend.train(
        data_yaml,
        cfg,
        resume_from=resume_from,
    )
    if cfg.training_outputs.checkpoints.dir is None:
        raise RuntimeError("training_outputs.checkpoints.dir was not resolved")
    contract = build_inference_contract(cfg, class_mapping=class_mapping)
    canonical = {
        "best": finalize_checkpoint(
            native.best,
            role="best",
            output_dir=cfg.training_outputs.checkpoints.dir,
            contract=contract,
        ),
        "last": finalize_checkpoint(
            native.last,
            role="last",
            output_dir=cfg.training_outputs.checkpoints.dir,
            contract=contract,
        ),
    }
    write_checkpoint_metadata(
        cfg.training_outputs.checkpoints.dir,
        contract=contract,
        checkpoints=canonical,
    )
    if cfg.training_outputs.metrics.enabled and cfg.training_outputs.metrics.dir is not None:
        native_csv = native.root / "results.csv"
        if native_csv.is_file():
            normalize_epoch_metrics(
                native_csv,
                cfg.training_outputs.metrics.dir / "metrics.csv",
            )
    if cfg.training_outputs.figures.enabled and cfg.training_outputs.figures.dir is not None:
        copy_native_figures(native.root, cfg.training_outputs.figures.dir)
    best_path, best_hash = canonical["best"]
    return best_path, best_hash
