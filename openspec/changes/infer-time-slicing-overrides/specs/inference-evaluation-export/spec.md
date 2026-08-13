## MODIFIED Requirements

### Requirement: Checkpoint contract reconstruction
Infer, eval, and export SHALL reconstruct model, class, target, and
model-coupled preprocessing defaults from `sati_inference_contract`, failing
before a forward pass when absent or invalid. Model-coupled preprocessing SHALL
mean the emitted slice size and the slice image format; the remaining slice
geometry SHALL be taken from the resolved configuration rather than the
contract. Verification of the contract's own `preprocessing_hash` SHALL be
performed against preprocessing reconstructed from the contract, so that a
corrupt or inconsistent contract fails rather than verifying against values the
command just assigned.

#### Scenario: Legacy checkpoint
- **WHEN** a checkpoint lacks the contract
- **THEN** the command requests a new canonical SATI checkpoint

#### Scenario: Inconsistent contract
- **WHEN** a contract's stored `preprocessing_hash` disagrees with its own stored
  preprocessing fields
- **THEN** verification fails before a forward pass

### Requirement: Recombination override precedence
Infer/eval SHALL accept changes to recombination and result policy, including the
merge method, the match metric, the matching mode, and either stage threshold,
and SHALL accept changes to runtime slice geometry, meaning the level policy, the
overlap ratio, and the maximum level count. Infer/eval SHALL reject changes to
model architecture, class mapping, target schema, and model-coupled
preprocessing. A rejected change SHALL raise a refusal naming the offending
field; no override of any slicing field SHALL be discarded without either taking
effect or raising.

#### Scenario: Post-hoc threshold choice
- **WHEN** the cross-level threshold is overridden
- **THEN** final recombination identity changes while compatibility identities
  remain fixed

#### Scenario: Post-hoc strategy choice
- **WHEN** the merge method or match metric is overridden at infer/eval time
- **THEN** the override is accepted, recorded on merge rows, and moves only the
  recombination identity

#### Scenario: Post-hoc pyramid choice
- **WHEN** the level policy, overlap ratio, or maximum level count is overridden at
  infer/eval time
- **THEN** the runtime pyramid is built from the overridden values, the effective
  preprocessing identity on the result rows differs from the checkpoint's
  `preprocessing_hash`, and the checkpoint's `preprocessing_hash` is unchanged

#### Scenario: Model-coupled preprocessing override refused
- **WHEN** the resolved emitted slice size or slice image format differs from the
  checkpoint contract's
- **THEN** the command raises a refusal naming that field before a forward pass

### Requirement: Exported model identity
Export SHALL compute `model_hash` from exported bytes, retain
`checkpoint_hash`, and place the model and portable `inference_config.yaml`
under `export_outputs.models.dir`. Export SHALL source slice geometry from the
checkpoint contract regardless of the resolved configuration, so that exported
preprocessing metadata always describes the trained pyramid. A slice-geometry
override supplied to `export` SHALL emit a warning naming the ignored fields.

#### Scenario: Two export formats
- **WHEN** one checkpoint is exported twice
- **THEN** exports share checkpoint identity and have independent model hashes

#### Scenario: Geometry override is inert but audible
- **WHEN** `export` resolves a level policy, overlap ratio, or maximum level count
  differing from the checkpoint contract's
- **THEN** the exported inference config records the contract's geometry and the
  command warns that the override was not applied

## ADDED Requirements

### Requirement: Pyramid divergence warning
Infer and eval SHALL compare the planned runtime level stack against the level
stack the checkpoint contract's preprocessing plans for the same frame, and SHALL
emit a warning naming both stacks whenever the two are not equal. Partial overlap
between the stacks SHALL NOT suppress the warning. The warning SHALL NOT stop the
run, and no divergence SHALL be grounds for refusal. Divergence SHALL be evaluated
on planned level scales rather than on level-policy fields, because distinct
policies can plan identical stacks and identical policies can plan distinct stacks
across differently-sized frames.

#### Scenario: Distant pyramid
- **WHEN** the runtime level stack shares no level scale with the trained stack
- **THEN** the command warns naming both stacks and proceeds to inference

#### Scenario: Partially overlapping pyramid
- **WHEN** a runtime stack shares one or more level scales with the trained stack
  but is not equal to it
- **THEN** the command warns naming both stacks and proceeds to inference

#### Scenario: Policy change that plans the same stack
- **WHEN** the runtime level policy differs from the trained one but plans an
  identical level stack for the frame
- **THEN** no divergence warning is emitted

#### Scenario: Divergence never refuses
- **WHEN** a runtime stack diverges from the trained stack by any amount
- **THEN** the run proceeds and writes results

### Requirement: Full-frame evaluation
Evaluation SHALL infer over a prepared split, score only full-resolution
`final_merge` rows, and write `results/validation_results.parquet`,
`results/_metadata.json`, and `metrics/metrics.json` under `eval_outputs.dir`.
Evaluation SHALL source split membership, ground truth, and class mapping from the
prepared product, and SHALL build its slice pyramid at runtime from the resolved
configuration rather than from prepared slice artifacts. A slice-geometry override
SHALL therefore require no re-preparation of the dataset.

#### Scenario: Evaluation completes
- **WHEN** evaluation succeeds
- **THEN** config, results, metrics, and lifecycle state are discoverable from
  the eval root

#### Scenario: Re-sliced evaluation without re-preparation
- **WHEN** eval runs with an overridden level policy against an existing prepared
  product
- **THEN** scoring proceeds against the same ground truth, and the prepared dataset
  identity on the result rows is unchanged while the effective preprocessing
  identity differs
