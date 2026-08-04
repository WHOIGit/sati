"""Collision-safe product/run initialization and provenance persistence."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sati.config_loader import ConfigResolution
from sati.config_schemas import RootConfig
from sati.errors import SatiError

CONFIG_DIR_NAME = "config"
RUN_MANIFEST_NAME = "run_manifest.json"


class LifecycleError(SatiError):
    """Run/product target or state is unsafe or incompatible."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=False) + "\n")


def validate_destructive_target(target: Path, *, cwd: Path | None = None) -> Path:
    target = target.resolve()
    cwd = (cwd or Path.cwd()).resolve()
    forbidden = {Path("/"), Path.home().resolve(), cwd}
    if target in forbidden:
        raise LifecycleError(f"refusing unsafe output target: {target}")
    if "{" in str(target) or "}" in str(target):
        raise LifecycleError(f"refusing unresolved output template: {target}")
    if len(target.parts) < 3:
        raise LifecycleError(f"refusing overly broad output target: {target}")
    return target


def prepare_target(
    target: Path,
    *,
    clobber: bool = False,
    allow_existing_empty: bool = True,
) -> Path:
    target = validate_destructive_target(target)
    if target.exists():
        contents = list(target.iterdir()) if target.is_dir() else [target]
        if contents:
            if not clobber:
                raise LifecycleError(
                    f"output directory is not empty: {target}; use --clobber to replace it"
                )
            shutil.rmtree(target)
        elif not allow_existing_empty and not clobber:
            raise LifecycleError(f"output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_config_bundle(
    target: Path,
    resolution: ConfigResolution,
    *,
    argv: list[str],
) -> Path:
    """Persist the authoritative resolved YAML plus provenance mirrors."""
    config_dir = target / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolution.config.model_dump(mode="json", exclude_none=False)
    atomic_write_text(
        config_dir / "composed.yaml",
        yaml.safe_dump(resolution.composed, sort_keys=False),
    )
    atomic_write_text(
        config_dir / "resolved.yaml",
        yaml.safe_dump(resolved, sort_keys=False),
    )
    atomic_write_json(config_dir / "resolved.json", resolved)
    atomic_write_text(config_dir / "cli.txt", " ".join(argv) + "\n")
    atomic_write_text(
        config_dir / "overrides.txt",
        "\n".join(resolution.source.overrides)
        + ("\n" if resolution.source.overrides else ""),
    )
    return config_dir / "resolved.yaml"


@dataclass
class RunLifecycle:
    root: Path
    manifest_name: str = RUN_MANIFEST_NAME
    manifest: dict[str, Any] | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_name

    def start(
        self,
        cfg: RootConfig,
        *,
        artifact_paths: dict[str, str] | None = None,
    ) -> None:
        now = _timestamp()
        self.manifest = {
            "schema_version": 1,
            "status": "running",
            "command": cfg.runtime.command,
            "started_at": now,
            "completed_at": None,
            "run_id": cfg.runtime.run_id,
            "config_id": cfg.runtime.config_id,
            "config_hash": cfg.runtime.config_hash,
            "dataset_hash": cfg.data.prepared.dataset_hash,
            "prepared_dataset_hash": cfg.data.prepared.prepared_dataset_hash,
            "prepared_dataset_id": cfg.data.prepared.prepared_dataset_id,
            "resolved_config": "config/resolved.yaml",
            "artifacts": artifact_paths or {},
            "failure": None,
        }
        atomic_write_json(self.manifest_path, self.manifest)

    def resume(self, cfg: RootConfig | None = None) -> None:
        if not self.manifest_path.is_file():
            raise LifecycleError(f"resume manifest is missing: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text())
        if cfg is not None:
            if cfg.runtime.command != "train":
                raise LifecycleError("resume is supported only for training runs")
            mismatches = [
                f"{name}: expected {expected!r}, observed {self.manifest.get(name)!r}"
                for name, expected in (
                    ("command", cfg.runtime.command),
                    ("config_hash", cfg.runtime.config_hash),
                    ("config_id", cfg.runtime.config_id),
                    ("run_id", cfg.runtime.run_id),
                )
                if self.manifest.get(name) != expected
            ]
            if mismatches:
                raise LifecycleError(
                    "resume configuration does not match the run manifest — "
                    + "; ".join(mismatches)
                )
        if self.manifest.get("status") not in {"running", "failed"}:
            raise LifecycleError(
                f"cannot resume lifecycle with status {self.manifest.get('status')!r}"
            )
        self.manifest["status"] = "running"
        self.manifest["completed_at"] = None
        self.manifest["failure"] = None
        atomic_write_json(self.manifest_path, self.manifest)

    def update_artifacts(self, artifacts: dict[str, str]) -> None:
        if self.manifest is None:
            raise LifecycleError("lifecycle has not started")
        self.manifest["artifacts"].update(artifacts)
        atomic_write_json(self.manifest_path, self.manifest)

    def complete(self, artifacts: dict[str, str] | None = None) -> None:
        if self.manifest is None:
            raise LifecycleError("lifecycle has not started")
        if artifacts:
            self.manifest["artifacts"].update(artifacts)
        self.manifest["status"] = "complete"
        self.manifest["completed_at"] = _timestamp()
        atomic_write_json(self.manifest_path, self.manifest)

    def fail(self, error: BaseException) -> None:
        if self.manifest is None:
            return
        self.manifest["status"] = "failed"
        self.manifest["completed_at"] = _timestamp()
        self.manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(self.manifest_path, self.manifest)


def active_output_dir(cfg: RootConfig) -> Path:
    if cfg.runtime.command is None:
        raise LifecycleError("config has no active runtime.command")
    block = cfg.output_for(cfg.runtime.command)
    if block.dir is None:
        raise LifecycleError("active output block has no resolved dir")
    return block.dir
