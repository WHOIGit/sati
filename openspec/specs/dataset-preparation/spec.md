# Dataset Preparation Specification

## Purpose

Current raw ingestion, reusable prepared products, canonical manifests, and
format conversion.

## Requirements

### Requirement: COCO and YOLO ingestion
The system SHALL ingest COCO or YOLO into canonical full-frame and annotation
records using top-left-origin, half-open `xyxy`.

#### Scenario: COCO sparse category IDs
- **WHEN** COCO uses sparse category identifiers
- **THEN** they are remapped to contiguous zero-based indices by source ID

### Requirement: User-authored splits
The system SHALL preserve source splits and support an optional TOML override
addressing frames by filename, unique stem, or canonical ID.

#### Scenario: Ambiguous split alias
- **WHEN** an entry resolves to multiple frames
- **THEN** preparation fails

### Requirement: Distinct raw and prepared data references
Config SHALL represent ingestible input under `data.source` and a reusable SATI
product under `data.prepared`; prepare/convert consume the former and
train/eval consume the latter.

#### Scenario: One experiment prepares and trains
- **WHEN** `data.prepared.uri` is omitted
- **THEN** it resolves to `prepare_outputs.dir`

### Requirement: Logical dataset identity
Preparation SHALL compute `dataset_hash` from canonical frame, annotation,
split, class-index, and source-identity information, excluding slicing,
emitted files, and pixels.

#### Scenario: Slicing changes
- **WHEN** only slicing policy changes
- **THEN** `dataset_hash` remains stable

### Requirement: Prepared dataset identity
Preparation SHALL compute `prepared_dataset_hash` from `dataset_hash` and the
complete versioned preparation recipe, deriving `prepared_dataset_id` from it.

#### Scenario: Target gate changes
- **WHEN** a preparation target gate changes
- **THEN** prepared identity changes while logical dataset identity does not

### Requirement: Materialized prepared dataset
`sati prepare` SHALL write to `prepare_outputs.dir`, including `config/`,
`manifests/`, a derived `yolo/` tree, and `prepare_manifest.json`.

#### Scenario: Default prepared output
- **WHEN** no concrete prepare directory is authored
- **THEN** it renders under `./.cache/prepared_datasets/<prepared-dataset-id>`

#### Scenario: Negative slice
- **WHEN** a training slice has no retained target
- **THEN** its image and an empty YOLO label file are written

### Requirement: Explicit-schema manifests
Preparation SHALL write canonical Parquet manifests column-by-column against
hand-written schemas plus a strict `_metadata.json`.

#### Scenario: Dropped projected annotation
- **WHEN** a projected annotation fails a target gate
- **THEN** its manifest row records the drop and it is omitted from labels

### Requirement: Prepared product completion manifest
A prepared product SHALL have an atomically updated `prepare_manifest.json`
with lifecycle state, identities, versions, artifacts, and counts; only
`complete` is reusable.

#### Scenario: Interrupted preparation
- **WHEN** preparation does not complete
- **THEN** consumers refuse the product

### Requirement: Verified prepared product reuse
`sati prepare` SHALL reuse a complete matching product; a mismatch or
incomplete target SHALL require `--clobber`.

#### Scenario: Repeat preparation
- **WHEN** the same product already exists complete
- **THEN** preparation succeeds without re-emitting pixels

### Requirement: Prepared-tree compatibility gate
Prepared metadata and `data.yaml` SHALL carry logical, prepared, and
preprocessing identities; training SHALL report expected and observed values
for any mismatch before loading a model.

#### Scenario: Wrong prepared product
- **WHEN** selected and expected identities differ
- **THEN** training refuses before model load

### Requirement: Format conversion
`sati convert` SHALL translate COCO/YOLO and canonical manifests without
slicing.

#### Scenario: Conversion to YOLO
- **WHEN** manifests are converted to YOLO
- **THEN** labels derive from canonical boxes and source images are linked
