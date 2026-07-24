# Model Training Specification

## Purpose

Current stock-Ultralytics backend boundary and materialized-dataset training
path.

## Requirements

### Requirement: Backend protocol
The system SHALL isolate model-framework operations behind a backend protocol
and SHALL provide a stock Ultralytics implementation without subclassing its
dataset, trainer, or loss.

#### Scenario: Core install
- **WHEN** SATI is imported without the `ultralytics` extra
- **THEN** non-model modules remain importable and a model operation reports a
  clear missing-extra error

### Requirement: Stock materialized training
`sati train` SHALL train against a SATI-prepared `yolo/data.yaml`, passing model
image size, epochs, batch size, seed, optional learning rate, optional device,
and explicitly authored YOLO augmentation values.

#### Scenario: Omitted augmentation value
- **WHEN** a `training.yolo_augment` field is omitted
- **THEN** SATI does not pass that argument and the Ultralytics default applies

### Requirement: Current training location
The Ultralytics backend SHALL write the training job under
`<output.run_dir>/train/` and SHALL return
`<output.run_dir>/train/weights/best.pt`.

#### Scenario: Missing best checkpoint
- **WHEN** Ultralytics returns but `train/weights/best.pt` does not exist
- **THEN** the training command fails

### Requirement: Checkpoint byte identity
After training, the system SHALL compute `checkpoint_hash` as SHA-256 over the
returned checkpoint bytes.

#### Scenario: Training completion output
- **WHEN** training succeeds
- **THEN** the CLI reports the checkpoint path and full checkpoint hash
