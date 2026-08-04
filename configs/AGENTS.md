# configs

## Purpose

Authored local Hydra config roots and supporting config inputs for SATI runs.

## Ownership

Owns `configs/experiment/` roots automatically selected by `experiment=...` from the repository CWD and supporting inputs such as split TOML files.

## Local Contracts

- Experiment YAML files are full config roots with `# @package _global_` and their own `defaults` list.
- Runnable experiments must validate against `sati.config_schemas.RootConfig`.
- Proposed-schema examples must say so in file comments and must not be presented as runnable until matching schema/loader support exists.
- Split TOML files use SATI split keys: `training`, `validation`, `holdout`; values are source image filenames, source image stems, or SATI fullframe ids as strings.

## Work Guidance

- Keep dataset-specific experiments under `configs/experiment/v1/<dataset-name>/`.
- Prefer relative paths from the repository root so configs remain portable within the checkout.

## Verification

- Runnable SATI experiment configs: compose from the repository root with `sati.config_loader.compose_and_resolve(...)`; no explicit config directory is required.
- Proposed-schema examples: review comments and referenced files; full validation waits for schema support.

## Child DOX Index

(none)
