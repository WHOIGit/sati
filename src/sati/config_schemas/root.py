"""Strict SATI configuration schemas.

The authored and resolved forms share one Pydantic tree. Fields marked
``ResolvedOnlyMarker`` may be absent from authored YAML and are populated by
the resolver. All filesystem paths use ``SatiPath`` so path canonicalization
and resolution remain type-driven.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SatiPathMarker:
    """A field whose value is a filesystem path."""


class HashDerivedMarker:
    """A generated field excluded from behavioral config hashes."""


class ResolvedOnlyMarker:
    """A field populated by resolution rather than authored YAML."""


SATI_PATH = SatiPathMarker()
HASH_DERIVED = HashDerivedMarker()
RESOLVED_ONLY = ResolvedOnlyMarker()

SatiPath = Annotated[Path, SATI_PATH]
ResolvedString = Annotated[str | None, HASH_DERIVED, RESOLVED_ONLY]
ResolvedPath = Annotated[SatiPath | None, HASH_DERIVED, RESOLVED_ONLY]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1)


class TaskConfig(StrictModel):
    # The coordinate convention (top-left, half-open xyxy) is a hard invariant,
    # captured as a versioned constant in target_schema_hash — not authored
    # config. Only the task type is a real selector (v1: object_detection).
    type: Literal["object_detection"] = "object_detection"


class RuntimeConfig(StrictModel):
    seed: int = 0
    run_id: Annotated[str, HASH_DERIVED] = "{coolname:noseed}"
    config_id: ResolvedString = None
    command: ResolvedString = None
    config_hash: ResolvedString = None


class RawDataSourceConfig(StrictModel):
    format: Literal["coco", "yolo"]
    uri: SatiPath
    split_file: SatiPath | None = None
    image_root: SatiPath | None = None
    dataset_id: str | None = None

    @model_validator(mode="after")
    def _image_root_is_coco_only(self) -> "RawDataSourceConfig":
        if self.image_root is not None and self.format != "coco":
            raise ValueError("data.source.image_root is only supported for format=coco")
        return self


class PreparedDataConfig(StrictModel):
    uri: SatiPath | None = None
    dataset_hash: ResolvedString = None
    prepared_dataset_hash: ResolvedString = None
    prepared_dataset_id: ResolvedString = None


class DataConfig(StrictModel):
    source: RawDataSourceConfig
    prepared: PreparedDataConfig = Field(default_factory=PreparedDataConfig)


class AutoLevelPolicyConfig(StrictModel):
    kind: Literal["auto"] = "auto"
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


class ModelArchitectureConfig(StrictModel):
    source: Literal["ultralytics"] = "ultralytics"
    name: str


class ModelWeightsConfig(StrictModel):
    source: Literal["none", "library", "checkpoint"] = "library"
    uri: str | None = None

    @model_validator(mode="after")
    def _validate_uri(self) -> "ModelWeightsConfig":
        if self.source == "checkpoint" and not self.uri:
            raise ValueError("model.weights.uri is required when source=checkpoint")
        if self.source == "none" and self.uri is not None:
            raise ValueError("model.weights.uri must be absent when source=none")
        return self


class ModelConfig(StrictModel):
    architecture: ModelArchitectureConfig
    weights: ModelWeightsConfig = Field(default_factory=ModelWeightsConfig)
    num_classes: int = Field(ge=1)
    imgsz: int | None = Field(default=None, ge=32)


class YoloAugmentConfig(StrictModel):
    """YOLO train-time augmentation knobs.

    SATI's default is **fully deactivated** augmentation: every knob defaults to
    its off-value, so training is deterministic unless an augmentation is
    explicitly requested. This deliberately breaks with Ultralytics' stock
    defaults (mosaic on, HSV/scale/erasing/flip active) — reproduce those by
    composing the packaged ``augmentation/ultralytics`` group. ``auto_augment``
    is classification-only in Ultralytics (a no-op for detection), so it has no
    off-value and stays ``None``.

    A per-knob ``null`` is still honored as "defer to Ultralytics' own default
    for this one knob" via :meth:`explicit_train_kwargs`.
    """

    hsv_h: float | None = Field(default=0.0, ge=0.0, le=1.0)
    hsv_s: float | None = Field(default=0.0, ge=0.0, le=1.0)
    hsv_v: float | None = Field(default=0.0, ge=0.0, le=1.0)
    translate: float | None = Field(default=0.0, ge=0.0, le=1.0)
    scale: float | None = Field(default=0.0, ge=0.0)
    flipud: float | None = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float | None = Field(default=0.0, ge=0.0, le=1.0)
    mosaic: float | None = Field(default=0.0, ge=0.0, le=1.0)
    close_mosaic: int | None = Field(default=0, ge=0)
    auto_augment: str | None = None
    erasing: float | None = Field(default=0.0, ge=0.0, le=1.0)

    def explicit_train_kwargs(self) -> dict:
        # Forward every knob that carries a concrete value. ``None`` means
        # "defer to Ultralytics' own default for this knob" — never pass it
        # through (Ultralytics rejects ``knob=None`` for typed knobs, and
        # ``auto_augment`` is a detection no-op). Filtering by value (not
        # ``model_fields_set``) survives the resolver's dump→revalidate
        # round-trip, which marks every field as explicitly set.
        dump = self.model_dump()
        return {name: value for name, value in dump.items() if value is not None}


class TrainingConfig(StrictModel):
    epochs: int = Field(default=100, ge=1)
    batch: int = Field(default=16, ge=1)
    lr: float | None = Field(default=None, gt=0.0)
    device: str | None = None
    # Early stopping. Ultralytics exposes a single integer knob: stop after this
    # many epochs without val-fitness improvement. ``None`` defers to
    # Ultralytics' own default (100). A concrete value is behaviorally relevant
    # and so contributes to ``config_hash`` (unlike ``None``, which drops out
    # under ``exclude_none``).
    patience: int | None = Field(default=None, ge=1)
    yolo_augment: YoloAugmentConfig = Field(default_factory=YoloAugmentConfig)


class RecombinationConfig(StrictModel):
    within_level_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_level_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    class_agnostic: bool = False


class ResultsConfig(StrictModel):
    cache_slice_predictions: bool = False


class OutputLeafConfig(StrictModel):
    enabled: bool = True
    dir: SatiPath | None = None


class OutputBlockConfig(StrictModel):
    dir: SatiPath | None = None
    dir_template: str | None = None

    @model_validator(mode="after")
    def _has_dir_source(self) -> "OutputBlockConfig":
        if self.dir is None and self.dir_template is None:
            raise ValueError("an output block requires dir or dir_template")
        return self


class PrepareOutputsConfig(OutputBlockConfig):
    dir_template: str | None = "./.cache/prepared_datasets/{data.prepared.prepared_dataset_id}"


class TrainingOutputsConfig(OutputBlockConfig):
    dir_template: str | None = "{experiment.name}/{timestamp}_{runtime.run_id}"
    checkpoints: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("checkpoints"))
    )
    metrics: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("metrics"))
    )
    figures: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(enabled=True, dir=Path("figures"))
    )
    backend: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("backend"))
    )


class EvalOutputsConfig(OutputBlockConfig):
    dir_template: str | None = "{experiment.name}/{timestamp}_{runtime.run_id}_eval"
    results: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("results"))
    )
    metrics: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("metrics"))
    )
    figures: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(enabled=False, dir=Path("figures"))
    )


class InferOutputsConfig(OutputBlockConfig):
    dir_template: str | None = "{experiment.name}/{timestamp}_{runtime.run_id}_infer"
    results: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("results"))
    )
    figures: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(enabled=False, dir=Path("figures"))
    )


class ExportOutputsConfig(OutputBlockConfig):
    dir_template: str | None = "{source_run_dir}/exports/{timestamp}_{runtime.run_id}"
    models: OutputLeafConfig = Field(
        default_factory=lambda: OutputLeafConfig(dir=Path("models"))
    )


CommandName = Literal["prepare", "convert", "train", "eval", "infer", "export"]


class RootConfig(StrictModel):
    experiment: ExperimentConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig
    slicing: SlicingConfig
    model: ModelConfig
    training: TrainingConfig
    recombination: RecombinationConfig
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    output_root: SatiPath = Path("./runs")
    prepare_outputs: PrepareOutputsConfig = Field(default_factory=PrepareOutputsConfig)
    training_outputs: TrainingOutputsConfig = Field(default_factory=TrainingOutputsConfig)
    eval_outputs: EvalOutputsConfig = Field(default_factory=EvalOutputsConfig)
    infer_outputs: InferOutputsConfig = Field(default_factory=InferOutputsConfig)
    export_outputs: ExportOutputsConfig = Field(default_factory=ExportOutputsConfig)

    def output_for(self, command: CommandName) -> OutputBlockConfig:
        attribute = {
            "prepare": "prepare_outputs",
            "convert": "prepare_outputs",
            "train": "training_outputs",
            "eval": "eval_outputs",
            "infer": "infer_outputs",
            "export": "export_outputs",
        }[command]
        return getattr(self, attribute)


def _annotation_has_marker(annotation: object, marker_type: type) -> bool:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if any(isinstance(marker, marker_type) for marker in args[1:]):
            return True
        return _annotation_has_marker(args[0], marker_type)
    return any(_annotation_has_marker(arg, marker_type) for arg in get_args(annotation))


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
    paths: list[tuple[str, ...]] = []
    for name, field in model_cls.model_fields.items():
        here = _prefix + (name,)
        if any(isinstance(marker, marker_type) for marker in field.metadata) or (
            _annotation_has_marker(field.annotation, marker_type)
        ):
            paths.append(here)
            continue
        for submodel in _model_classes_in(field.annotation):
            paths.extend(annotated_field_paths(submodel, marker_type, here))
    return paths


def _annotation_contains_type(annotation: object, target: type) -> bool:
    if annotation is target:
        return True
    if get_origin(annotation) is Annotated:
        return _annotation_contains_type(get_args(annotation)[0], target)
    return any(_annotation_contains_type(arg, target) for arg in get_args(annotation))


def bare_path_fields(
    model_cls: type[BaseModel],
    _prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    offenders: list[tuple[str, ...]] = []
    for name, field in model_cls.model_fields.items():
        here = _prefix + (name,)
        has_path = _annotation_contains_type(field.annotation, Path)
        has_marker = any(isinstance(marker, SatiPathMarker) for marker in field.metadata) or (
            _annotation_has_marker(field.annotation, SatiPathMarker)
        )
        if has_path and not has_marker:
            offenders.append(here)
        for submodel in _model_classes_in(field.annotation):
            offenders.extend(bare_path_fields(submodel, here))
    return offenders
