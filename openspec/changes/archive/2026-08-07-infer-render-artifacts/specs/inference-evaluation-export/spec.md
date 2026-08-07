## MODIFIED Requirements

### Requirement: Inference output envelope
Inference SHALL write `results/predictions.parquet` and `_metadata.json` under
`infer_outputs.dir`, with config provenance and a lifecycle manifest. It MAY
additionally write detection overlays under an `artifacts/` directory in the same
run. Overlay emission SHALL default to enabled for inference and disabled for
evaluation, because evaluation covers a whole split and would otherwise render
every frame's full level stack unasked.

#### Scenario: Inference completes
- **WHEN** ad-hoc inference succeeds
- **THEN** it does not use the validation-table filename

#### Scenario: Uncombined overlays without slice caching
- **WHEN** slice-prediction caching is false and rendering is enabled
- **THEN** the uncombined per-level overlays are still written, because they are
  rendered from in-memory predictions rather than from the results table

#### Scenario: Evaluation over a split
- **WHEN** evaluation runs without explicitly enabling rendering
- **THEN** no overlay artifacts are written
