"""Pydantic contract for all SATI options and I/O typing (DESIGN-DOC-v1 §3).

Single source of truth for valid options. Strict everywhere: ``extra="forbid"``,
absent = unsupported. Zero Hydra imports.

Path fields must use the ``SatiPath`` annotated alias — a bare ``pathlib.Path``
in a schema is a bug (the hasher and the resolver both discover path fields by
annotation, never by name; DESIGN-DOC-v1 §3.2, §4.7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SatiPathMarker:
    """Annotation marker: field holds a filesystem path.

    The resolver absolutizes marked fields; the config hasher replaces their
    values with the ``"<path>"`` sentinel so no absolute path reaches the hash.
    """


class HashDerivedMarker:
    """Annotation marker: field is derived-only and excluded from config_hash."""


SATI_PATH = SatiPathMarker()
HASH_DERIVED = HashDerivedMarker()

SatiPath = Annotated[Path, SATI_PATH]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(StrictModel):
    source_format: Literal["coco", "yolo"]
    source_path: SatiPath
    split_file: SatiPath | None = None
    # COCO only: root that file_name entries resolve against; default is each
    # annotation file's own directory. YOLO layouts carry their own roots.
    image_root: SatiPath | None = None

    @model_validator(mode="after")
    def _image_root_is_coco_only(self) -> "DataConfig":
        if self.image_root is not None and self.source_format != "coco":
            raise ValueError("data.image_root is only supported for source_format=coco")
        return self


class AutoLevelPolicyConfig(StrictModel):
    kind: Literal["auto"] = "auto"
    # each policy owns its own factor: a knob that a policy ignores must not exist
    downsample_factor: float = Field(default=2.0, gt=1.0)


class StartThenGeometricLevelPolicyConfig(StrictModel):
    kind: Literal["start_then_geometric"] = "start_then_geometric"
    start_scale: float = Field(gt=0.0, le=1.0)
    subsequent_downsample_factor: float = Field(gt=1.0)
    stop_when_long_side_lte_nominal: bool = True


LevelPolicyConfig = Annotated[
    AutoLevelPolicyConfig | StartThenGeometricLevelPolicyConfig,
    Field(discriminator="kind"),
]


class SlicingConfig(StrictModel):
    level_policy: LevelPolicyConfig = Field(default_factory=AutoLevelPolicyConfig)
    nominal_slice_size: int = Field(default=1024, ge=1)
    overlap_ratio: float = Field(default=0.10, ge=0.0, lt=1.0)
    max_levels: int | None = Field(default=None, ge=1)
    min_area_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    size_gate_px: int = Field(default=16, ge=0)
    image_format: Literal["png", "jpeg"] = "png"

    @model_validator(mode="after")
    def _bounded_level_policy(self) -> "SlicingConfig":
        if (
            isinstance(self.level_policy, StartThenGeometricLevelPolicyConfig)
            and not self.level_policy.stop_when_long_side_lte_nominal
            and self.max_levels is None
        ):
            raise ValueError(
                "slicing.max_levels is required when level_policy does not stop "
                "at nominal size"
            )
        return self


class ModelConfig(StrictModel):
    arch: str
    # weights may name a hub asset ("yolo26n.pt") or a local file; the backend
    # interprets it, so it stays a plain string and participates in the hash.
    # Actual weight identity is carried by checkpoint_hash, not here.
    weights: str | None = None
    num_classes: int = Field(ge=1)
    imgsz: int | None = Field(default=None, ge=32)  # None → nominal_slice_size at resolve


class YoloAugmentConfig(StrictModel):
    hsv_h: float | None = Field(default=None, ge=0.0, le=1.0)
    hsv_s: float | None = Field(default=None, ge=0.0, le=1.0)
    hsv_v: float | None = Field(default=None, ge=0.0, le=1.0)
    translate: float | None = Field(default=None, ge=0.0, le=1.0)
    scale: float | None = Field(default=None, ge=0.0)
    flipud: float | None = Field(default=None, ge=0.0, le=1.0)
    fliplr: float | None = Field(default=None, ge=0.0, le=1.0)
    mosaic: float | None = Field(default=None, ge=0.0, le=1.0)
    close_mosaic: int | None = Field(default=None, ge=0)
    auto_augment: str | None = None
    erasing: float | None = Field(default=None, ge=0.0, le=1.0)

    def explicit_train_kwargs(self) -> dict:
        dump = self.model_dump()
        return {name: dump[name] for name in self.model_fields_set}


class TrainingConfig(StrictModel):
    epochs: int = Field(default=100, ge=1)
    batch: int = Field(default=16, ge=1)
    lr: float | None = Field(default=None, gt=0.0)
    seed: int = 0
    device: str | None = None
    yolo_augment: YoloAugmentConfig = Field(default_factory=YoloAugmentConfig)


class RecombinationConfig(StrictModel):
    within_level_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_level_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    class_agnostic: bool = False


class ResultsConfig(StrictModel):
    cache_slice_predictions: bool = False


class OutputConfig(StrictModel):
    run_dir: SatiPath


class RootConfig(StrictModel):
    data: DataConfig
    slicing: SlicingConfig
    model: ModelConfig
    training: TrainingConfig
    recombination: RecombinationConfig
    results: ResultsConfig = ResultsConfig()
    output: OutputConfig
    run_id: Annotated[str | None, HASH_DERIVED] = None


def _annotation_has_marker(annotation: object, marker_type: type) -> bool:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if any(isinstance(m, marker_type) for m in args[1:]):
            return True
        return _annotation_has_marker(args[0], marker_type)
    return any(_annotation_has_marker(a, marker_type) for a in get_args(annotation))


def _model_classes_in(annotation: object) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    out: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        out.extend(_model_classes_in(arg))
    return out


def annotated_field_paths(
    model_cls: type[BaseModel],
    marker_type: type,
    _prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """Dotted paths (as tuples) of every field carrying ``marker_type``, depth-first.

    Pydantic hoists top-level ``Annotated`` metadata into ``FieldInfo.metadata``;
    metadata nested inside unions (``SatiPath | None``) stays in the annotation,
    so both places are searched.
    """
    paths: list[tuple[str, ...]] = []
    for name, field in model_cls.model_fields.items():
        here = _prefix + (name,)
        if any(isinstance(m, marker_type) for m in field.metadata) or _annotation_has_marker(
            field.annotation, marker_type
        ):
            paths.append(here)
            continue
        for sub in _model_classes_in(field.annotation):
            paths.extend(annotated_field_paths(sub, marker_type, here))
    return paths


def _annotation_contains_type(annotation: object, target: type) -> bool:
    if annotation is target:
        return True
    if get_origin(annotation) is Annotated:
        return _annotation_contains_type(get_args(annotation)[0], target)
    return any(_annotation_contains_type(a, target) for a in get_args(annotation))


def bare_path_fields(
    model_cls: type[BaseModel],
    _prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    """Fields whose type involves ``Path`` without the ``SatiPath`` marker — bugs.

    Used by the schema guard test (DESIGN-DOC-v1 §4.7).
    """
    offenders: list[tuple[str, ...]] = []
    for name, field in model_cls.model_fields.items():
        here = _prefix + (name,)
        has_path = _annotation_contains_type(field.annotation, Path)
        has_marker = any(
            isinstance(m, SatiPathMarker) for m in field.metadata
        ) or _annotation_has_marker(field.annotation, SatiPathMarker)
        if has_path and not has_marker:
            offenders.append(here)
        for sub in _model_classes_in(field.annotation):
            offenders.extend(bare_path_fields(sub, here))
    return offenders
