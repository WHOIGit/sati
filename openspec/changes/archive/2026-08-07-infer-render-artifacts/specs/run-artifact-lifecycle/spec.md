## MODIFIED Requirements

### Requirement: Lifecycle manifest
Each invocation SHALL atomically track command, state, timestamps, identities,
hashes, artifact paths, and failures. Every declared output leaf the invocation
actually wrote SHALL appear among those artifact paths, so an overlay tree is
discoverable from the manifest rather than being an undeclared side effect.

#### Scenario: Failed invocation
- **WHEN** work raises after initialization
- **THEN** the manifest becomes `failed` and provenance remains

#### Scenario: Overlays written alongside results
- **WHEN** inference writes both its results table and its overlay tree
- **THEN** the manifest records both leaves, and the overlay leaf lives under the
  same run root and is therefore covered by the same collision and clobber rules
