# Slicing and Recombination Specification

## Purpose

Current multi-resolution frame planning, slice emission, affine provenance,
target projection, and two-stage detection recombination.

## Requirements

### Requirement: Resolution stack policies
The system SHALL support `auto` and `start_then_geometric` level policies, and
level index zero SHALL mean full resolution whenever full resolution is
present.

#### Scenario: Stack starts below full resolution
- **WHEN** `start_then_geometric.start_scale` is less than one
- **THEN** the first emitted level is numbered one

### Requirement: Complete deterministic tiling
Every planned level SHALL be fully covered by deterministic overlapping
windows, with the last window on each non-short axis clamped to the far edge.

#### Scenario: Axis shorter than nominal
- **WHEN** one level axis is shorter than the nominal slice size
- **THEN** one span covers that axis and the emitted pixels are stretched to
  nominal size

### Requirement: Invertible slice provenance
Every emitted slice SHALL record per-axis affine transforms from emitted-slice
coordinates to level and full-resolution coordinates.

#### Scenario: Round-trip box
- **WHEN** a full-resolution box is projected into a retained slice and mapped
  back through the recorded affine
- **THEN** the coordinates agree within floating-point tolerance

### Requirement: Sharp-source stretch
The system SHALL source stretched windows from the finest available level
whose mapped window is at least the emitted size on both axes, falling back to
the full-resolution frame.

#### Scenario: Whole coarse level is stretched
- **WHEN** a coarse whole-level window is smaller than nominal
- **THEN** the emitted slice is not upscaled from that coarse level's pixels

### Requirement: Two-stage NMS recombination
The system SHALL confidence-filter and NMS detections within each level, then
map winners to full resolution and NMS across levels, with optional
class-agnostic matching.

#### Scenario: Final detection
- **WHEN** a detection survives cross-level NMS
- **THEN** its final box is in full-frame full-resolution coordinates and
  records its contributing source levels
