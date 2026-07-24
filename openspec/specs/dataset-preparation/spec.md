# Dataset Preparation Specification

## Purpose

Current COCO/YOLO ingestion, user-authored splits, materialized slice store,
canonical manifests, and format conversion.

## Requirements

### Requirement: COCO and YOLO ingestion
The system SHALL ingest COCO or YOLO source datasets into canonical full-frame
and annotation records using top-left-origin, half-open `xyxy` coordinates.

#### Scenario: COCO sparse category IDs
- **WHEN** a COCO source uses sparse category identifiers
- **THEN** ingestion remaps them to contiguous zero-based class indices ordered
  by the source category ID

### Requirement: User-authored splits
The system SHALL preserve source-provided splits and SHALL allow an optional
TOML split file whose `training`, `validation`, and `holdout` entries identify
frames by source filename, unique stem, or canonical full-frame ID.

#### Scenario: Ambiguous split alias
- **WHEN** a TOML entry resolves to more than one source frame
- **THEN** preparation fails rather than choosing one

### Requirement: Materialized prepared dataset
`sati prepare` SHALL write its prepared dataset to `output.run_dir`, including
`manifests/` and a derived `yolo/` tree of emitted images, label files, and
`data.yaml`.

#### Scenario: Negative slice
- **WHEN** an emitted training slice contains no retained target
- **THEN** its image is written and its matching YOLO label file is empty

### Requirement: Explicit-schema manifests
Preparation SHALL write `fullframes.parquet`, `annotations.parquet`,
`slices.parquet`, and `slice_annotations.parquet` column-by-column against
hand-written PyArrow schemas, plus a strict `_metadata.json` sidecar.

#### Scenario: Dropped projected annotation
- **WHEN** a projected annotation fails the size or visible-area gate
- **THEN** it remains in `slice_annotations.parquet` with the relevant drop
  flag and is omitted from the YOLO label file

### Requirement: Prepared-tree compatibility gate
The generated `yolo/data.yaml` SHALL carry the dataset-level
`preprocessing_hash`, and training SHALL refuse a prepared tree whose embedded
hash differs from the composed slicing configuration.

#### Scenario: Slicing configuration changed after preparation
- **WHEN** training selects a prepared tree but changes overlap, level policy,
  nominal slice size, or stretch policy
- **THEN** training fails before loading the model

### Requirement: Format conversion
`sati convert` SHALL convert COCO or YOLO sources to canonical manifests and
canonical manifests back to COCO or YOLO without performing slicing.

#### Scenario: Conversion to YOLO
- **WHEN** canonical manifests are converted to YOLO
- **THEN** labels are derived from canonical boxes and source images are linked
  into the exported tree
