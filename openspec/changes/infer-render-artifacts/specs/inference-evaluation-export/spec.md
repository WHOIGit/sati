## MODIFIED Requirements

### Requirement: Inference output envelope
Inference SHALL write its tall results table under the active `results/` directory,
and MAY additionally write detection overlays under an `artifacts/` directory in
the same run. Overlay emission SHALL default to enabled for inference and disabled
for evaluation, because evaluation covers a whole split and would otherwise render
every frame's full level stack unasked.

#### Scenario: Slice cache disabled
- **WHEN** slice-prediction caching is false
- **THEN** tall results omit raw slice rows and retain merged/frame rows

#### Scenario: Uncombined overlays without slice caching
- **WHEN** slice-prediction caching is false and rendering is enabled
- **THEN** the uncombined per-level overlays are still written, because they are
  rendered from in-memory predictions rather than from the results table

#### Scenario: Evaluation over a split
- **WHEN** evaluation runs without explicitly enabling rendering
- **THEN** no overlay artifacts are written
