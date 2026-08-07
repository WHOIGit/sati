## ADDED Requirements

### Requirement: Pipeline-state artifact tree
Inference SHALL be able to write detection overlays into an `artifacts/` output
leaf laid out as `artifacts/<image-stem>/fullframes/`, containing one image per
recombination pipeline state at two scopes: per level, uncombined and combined; and
stacked across all levels on one canvas, uncombined and combined; plus the final
full-resolution detections. Level-indexed images SHALL use the frame's realized
level indices, so a stack that starts below full resolution emits no level-zero
image.

#### Scenario: Three-level frame
- **WHEN** a frame's realized stack has levels one through three
- **THEN** the tree contains a raw and a combined image for each of those three
  levels, plus a raw stacked image, a combined stacked image, and a final image

#### Scenario: Stack starts below full resolution
- **WHEN** the first realized level is numbered one
- **THEN** no level-zero image is written

### Requirement: Each artifact depicts one pipeline state
The uncombined per-level image SHALL show slice predictions at or above the
configured confidence threshold, mapped into that level's coordinates, before
within-level combining. The combined per-level image SHALL show that level's
within-level winners. The uncombined stacked image SHALL show every level's
confidence-filtered slice predictions mapped to full resolution. The combined
stacked image SHALL show every level's within-level winners mapped to full
resolution — the input to cross-level combining. The final image SHALL show the
cross-level winners. Both stacked images SHALL distinguish boxes by contributing
level.

#### Scenario: Attributing a lost box
- **WHEN** a detection appears in a level's uncombined image but not in that
  level's combined image
- **THEN** within-level combining removed it, and no other stage is implicated

#### Scenario: Stacked images are distinct states
- **WHEN** both stacked images are written
- **THEN** the uncombined one contains every level's confidence-filtered slice
  predictions and the combined one contains every level's within-level winners

### Requirement: Artifacts are drawn on their own state's canvas
Level-indexed images SHALL be drawn on that level's realized pixel dimensions, and
the stacked and final images on the full-resolution frame. Box geometry SHALL be
expressed in the coordinate space of its own canvas, so no artifact requires a
scale correction applied to its boxes.

#### Scenario: Level canvas dimensions
- **WHEN** a level's realized dimensions are smaller than full resolution
- **THEN** that level's images have those dimensions and its boxes are in level
  coordinates

### Requirement: Rendering moves no identity
Rendering SHALL NOT alter any detection, metric, result row, or compatibility
identity, and its style configuration SHALL be referenced by a portable path so
that restyling an overlay leaves every behavioral hash unchanged.

#### Scenario: Restyled overlay
- **WHEN** two runs differ only in overlay style
- **THEN** every hash and every result row is identical

#### Scenario: Rendering disabled
- **WHEN** rendering is disabled
- **THEN** results and identities are unchanged from an enabled run

### Requirement: Rendering is an optional capability
Rendering SHALL be an optional install. Preparation, training, evaluation, and
inference without rendering SHALL succeed in an environment lacking the rendering
dependencies, and requesting rendering without them SHALL fail with a
configuration error naming the missing extra rather than an import failure at
startup.

#### Scenario: Rendering requested without the extra installed
- **WHEN** rendering is enabled and the rendering dependencies are absent
- **THEN** the command fails with a configuration error naming the required extra

#### Scenario: Inference without rendering
- **WHEN** rendering is disabled and the rendering dependencies are absent
- **THEN** inference completes and writes results normally

### Requirement: Style is delegated, structure is owned
The system SHALL own artifact structure — which images exist, their layout, naming,
canvas, level selection, and the confidence filter — and SHALL delegate visual
style, label templating, and output encoding to the rendering library's own
configuration, overriding only the input document, image root, and output path.

#### Scenario: Unknown style key
- **WHEN** the referenced style configuration contains a key the rendering library
  rejects
- **THEN** the rendering library's validation error surfaces unchanged rather than
  being re-validated
