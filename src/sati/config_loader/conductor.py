"""Compose → validate → command-aware resolve, with no filesystem writes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from sati.config_loader.compositor import ComposedConfig, compose_config
from sati.config_loader.resolver import ResolutionResult, resolve_config, validate_resolved_config
from sati.config_schemas import CommandName, ResolvedOnlyMarker, RootConfig, annotated_field_paths


@dataclass(frozen=True)
class ConfigResolution:
    config: RootConfig
    composed: dict[str, Any]
    source: ComposedConfig
    warnings: tuple[str, ...] = ()


def _reject_authored_resolved_fields(container: dict[str, Any]) -> None:
    for path in annotated_field_paths(RootConfig, ResolvedOnlyMarker):
        node: Any = container
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, dict) and path[-1] in node and node[path[-1]] is not None:
            raise ValueError(f"{'.'.join(path)} is resolved-only")


def resolve_configuration(
    overrides: Sequence[str] = (),
    *,
    command: CommandName,
    config_dirs: Sequence[Path | str] = (),
    config_file: Path | str | None = None,
    resolved_config_file: Path | str | None = None,
    base_dir: Path | None = None,
    source_run_dir: Path | None = None,
    for_inspection: bool = False,
) -> ConfigResolution:
    composed = compose_config(
        overrides,
        config_dirs=[Path(path) for path in config_dirs],
        config_file=Path(config_file) if config_file is not None else None,
        resolved_config_file=(
            Path(resolved_config_file) if resolved_config_file is not None else None
        ),
    )
    container = OmegaConf.to_container(
        composed.config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(container, dict):
        raise ValueError("composed config root must be a mapping")
    if composed.resolved_source:
        cfg = validate_resolved_config(RootConfig.model_validate(container), command=command)
        return ConfigResolution(config=cfg, composed=container, source=composed)

    _reject_authored_resolved_fields(container)
    authored = RootConfig.model_validate(container)
    resolved: ResolutionResult = resolve_config(
        authored,
        command=command,
        cwd=base_dir,
        source_run_dir=source_run_dir,
        for_inspection=for_inspection,
    )
    return ConfigResolution(
        config=resolved.config,
        composed=container,
        source=composed,
        warnings=resolved.warnings,
    )


def compose_and_resolve(
    overrides: Sequence[str] = (),
    *,
    command: CommandName = "prepare",
    config_dirs: Sequence[Path | str] = (),
    config_file: Path | str | None = None,
    resolved_config: Path | str | None = None,
    base_dir: Path | None = None,
    run_id: str | None = None,
    write: bool = False,
) -> RootConfig:
    """Compatibility convenience returning only the typed resolved config.

    ``write`` is accepted only to make the clean break explicit: config
    resolution never writes. Mutating commands persist through the lifecycle
    layer after collision checks.
    """
    if write:
        raise ValueError("config resolution is side-effect-free; use the lifecycle writer")
    extra = list(overrides)
    if run_id is not None:
        extra.append(f"runtime.run_id={run_id}")
    return resolve_configuration(
        extra,
        command=command,
        config_dirs=config_dirs,
        config_file=config_file,
        resolved_config_file=resolved_config,
        base_dir=base_dir,
    ).config
