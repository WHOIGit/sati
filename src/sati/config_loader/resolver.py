"""Command-aware derived identity and output-path resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sati.config_schemas import (
    CommandName,
    OutputBlockConfig,
    RootConfig,
    SatiPathMarker,
    annotated_field_paths,
)
from sati.hashing import config_hash, config_id, deterministic_id, fresh_run_id

_TOKEN_RE = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class ResolutionResult:
    config: RootConfig
    warnings: tuple[str, ...] = ()


def _get(mapping: dict[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def render_template(
    template: str,
    values: dict[str, Any],
    *,
    timestamp: datetime | None = None,
) -> str:
    now = timestamp or datetime.now()

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "timestamp":
            return now.strftime("%Y-%m-%d_%H-%M-%S")
        if token == "coolname:noseed":
            return fresh_run_id()
        if token.startswith("coolname:"):
            source = token.split(":", 1)[1]
            try:
                value = str(_get(values, source))
            except KeyError:
                value = source
            return deterministic_id(source, value)
        path, separator, fmt = token.partition(":")
        value = _get(values, path)
        return format(value, fmt) if separator else str(value)

    rendered = _TOKEN_RE.sub(replace, template)
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"unresolved output template: {template}")
    return rendered


def _resolve_path(value: str | Path, parent: Path, cwd: Path) -> Path:
    text = str(value)
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if text == "." or text.startswith("./"):
        return (cwd / path).resolve()
    return (parent / path).resolve()


def _absolutize_input_paths(cfg: RootConfig, cwd: Path) -> None:
    output_prefixes = {
        "output_root",
        "prepare_outputs",
        "training_outputs",
        "eval_outputs",
        "infer_outputs",
        "export_outputs",
    }
    for path in annotated_field_paths(RootConfig, SatiPathMarker):
        if path[0] in output_prefixes:
            continue
        node: Any = cfg
        for part in path[:-1]:
            node = getattr(node, part)
        value = getattr(node, path[-1])
        if value is not None and not value.is_absolute():
            setattr(node, path[-1], (cwd / value).resolve())


def _resolve_leaf_dirs(block: OutputBlockConfig, run_dir: Path, cwd: Path) -> None:
    for name in type(block).model_fields:
        child = getattr(block, name)
        if not hasattr(child, "enabled") or not hasattr(child, "dir"):
            continue
        if child.dir is not None:
            child.dir = _resolve_path(child.dir, run_dir, cwd)


def resolve_config(
    authored: RootConfig,
    *,
    command: CommandName,
    cwd: Path | None = None,
    source_run_dir: Path | None = None,
) -> ResolutionResult:
    cwd = (cwd or Path.cwd()).resolve()
    cfg = authored.model_copy(deep=True)
    _absolutize_input_paths(cfg, cwd)
    if cfg.model.imgsz is None:
        cfg.model.imgsz = cfg.slicing.nominal_slice_size

    if command == "prepare" or (
        command in {"train", "eval"} and cfg.data.prepared.uri is None
    ):
        from sati.dataset_identity import populate_source_identities

        populate_source_identities(cfg)
    elif command in {"train", "eval"} and cfg.data.prepared.uri is not None:
        from sati.hashing import prepared_dataset_id
        from sati.manifests import read_metadata_sidecar

        metadata_dir = cfg.data.prepared.uri / "manifests"
        if metadata_dir.is_dir():
            metadata = read_metadata_sidecar(metadata_dir)
            for name in ("dataset_hash", "prepared_dataset_hash"):
                if name in metadata.hashes:
                    setattr(cfg.data.prepared, name, metadata.hashes[name].value)
            # The ID is a pure function of the prepared hash; derive it so a
            # reused product carries the same identity `prepare` recorded.
            if cfg.data.prepared.prepared_dataset_hash is not None:
                cfg.data.prepared.prepared_dataset_id = prepared_dataset_id(
                    cfg.data.prepared.prepared_dataset_hash
                )

    values = cfg.model_dump(mode="json", exclude_none=True)
    run_template = cfg.runtime.run_id or "{coolname:noseed}"
    if "{" in run_template:
        cfg.runtime.run_id = render_template(run_template, values)
    cfg.runtime.command = command
    digest = config_hash(cfg, command)
    cfg.runtime.config_hash = digest
    cfg.runtime.config_id = config_id(digest)

    output_root = _resolve_path(cfg.output_root, cwd, cwd)
    cfg.output_root = output_root
    values = cfg.model_dump(mode="json", exclude_none=True)
    values["source_run_dir"] = (
        str(source_run_dir.resolve()) if source_run_dir else str(output_root)
    )
    if command in {"train", "eval"} and cfg.data.prepared.uri is None:
        prepare = cfg.prepare_outputs
        if prepare.dir is not None:
            prepare.dir = _resolve_path(prepare.dir, output_root, cwd)
        elif prepare.dir_template is not None:
            prepare.dir = _resolve_path(
                render_template(prepare.dir_template, values),
                output_root,
                cwd,
            )
        else:
            raise ValueError("prepare_outputs requires dir or dir_template")
        cfg.data.prepared.uri = prepare.dir
        values = cfg.model_dump(mode="json", exclude_none=True)
        values["source_run_dir"] = (
            str(source_run_dir.resolve()) if source_run_dir else str(output_root)
        )
    block = cfg.output_for(command)
    if block.dir is not None:
        block.dir = _resolve_path(block.dir, output_root, cwd)
    elif block.dir_template is not None:
        try:
            rendered = render_template(block.dir_template, values)
        except KeyError as error:
            raise ValueError(
                f"{command} output template needs unresolved value {error.args[0]!r}"
            ) from error
        block.dir = _resolve_path(rendered, output_root, cwd)
    else:
        raise ValueError(f"{command} output block requires dir or dir_template")
    _resolve_leaf_dirs(block, block.dir, cwd)
    return ResolutionResult(config=RootConfig.model_validate(cfg.model_dump()), warnings=())


def validate_resolved_config(cfg: RootConfig, *, command: CommandName | None = None) -> RootConfig:
    if cfg.runtime.command is None or cfg.runtime.config_hash is None or cfg.runtime.config_id is None:
        raise ValueError("resolved config is missing runtime command/hash/ID fields")
    if command is not None and cfg.runtime.command != command:
        raise ValueError(
            f"resolved config command is {cfg.runtime.command!r}, not {command!r}"
        )
    active = cfg.output_for(cfg.runtime.command)
    if active.dir is None or not active.dir.is_absolute():
        raise ValueError("resolved config active output directory must be absolute")
    if config_hash(cfg, cfg.runtime.command) != cfg.runtime.config_hash:
        raise ValueError("resolved config_hash does not match its authoritative config")
    return cfg
