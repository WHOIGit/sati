"""Config-first SATI command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Callable, Optional, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from sati.checkpoints import apply_inference_contract
from sati.config_loader import ConfigResolution, resolve_configuration
from sati.config_schemas import CommandName
from sati.errors import SatiError
from sati.lifecycle import (
    RunLifecycle,
    active_output_dir,
    atomic_write_json,
    prepare_target,
    write_config_bundle,
)

app = typer.Typer(name="sati", no_args_is_help=True, add_completion=False)
console = Console()

Overrides = Annotated[
    Optional[list[str]],
    typer.Argument(help="Hydra overrides (dash-free key=value tokens)"),
]
ConfigDirs = Annotated[
    Optional[list[Path]],
    typer.Option("--config-dir", help="additional config directory; repeatable"),
]
ConfigOpt = Annotated[
    Optional[Path],
    typer.Option("--config", help="authored Hydra root YAML"),
]
ResolvedOpt = Annotated[
    Optional[Path],
    typer.Option("--resolved-config", help="authoritative resolved YAML for exact replay"),
]
ClobberOpt = Annotated[
    bool,
    typer.Option("--clobber", help="replace only the resolved concrete output directory"),
]

T = TypeVar("T")


def _validate_override_tokens(overrides: list[str]) -> None:
    for token in overrides:
        if token.startswith("-"):
            raise typer.BadParameter(
                f"unknown dash-prefixed option {token!r}; config overrides are dash-free"
            )


def _resolve(
    command: CommandName,
    overrides: list[str] | None,
    config_dirs: list[Path] | None,
    config_file: Path | None,
    resolved_config: Path | None,
    *,
    source_run_dir: Path | None = None,
) -> ConfigResolution:
    tokens = overrides or []
    _validate_override_tokens(tokens)
    return resolve_configuration(
        tokens,
        command=command,
        config_dirs=config_dirs or (),
        config_file=config_file,
        resolved_config_file=resolved_config,
        source_run_dir=source_run_dir,
    )


def _rel(path: Path, root: Path) -> str:
    """Run-root-relative path for the lifecycle manifest; absolute if outside."""
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run_lifecycle(
    resolution: ConfigResolution,
    *,
    clobber: bool,
    work: Callable[[], T],
    manifest_name: str = "run_manifest.json",
    artifacts_from: Callable[[T, Path], dict[str, str]] | None = None,
) -> T:
    root = active_output_dir(resolution.config)
    prepare_target(root, clobber=clobber)
    write_config_bundle(root, resolution, argv=sys.argv)
    lifecycle = RunLifecycle(root, manifest_name=manifest_name)
    lifecycle.start(resolution.config)
    try:
        result = work()
    except BaseException as error:
        lifecycle.fail(error)
        raise
    artifacts = artifacts_from(result, root) if artifacts_from is not None else None
    lifecycle.complete(artifacts)
    return result


def _prepared_product_reusable(resolution: ConfigResolution) -> bool:
    cfg = resolution.config
    root = active_output_dir(cfg)
    manifest_path = root / "prepare_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("prepared_dataset_hash")
        != cfg.data.prepared.prepared_dataset_hash
    ):
        return False
    required = (
        root / "manifests" / "_metadata.json",
        root / "manifests" / "slices.parquet",
        root / "yolo" / "data.yaml",
        root / "config" / "resolved.yaml",
    )
    return all(path.is_file() for path in required)


@app.command()
def prepare(
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
) -> None:
    """Ingest raw COCO/YOLO data and build a reusable prepared product."""
    from sati.materialize import execute_prepare

    resolution = _resolve("prepare", overrides, config_dir, config, resolved_config)
    if not clobber and _prepared_product_reusable(resolution):
        console.print(f"prepared product reused: {active_output_dir(resolution.config)}")
        return

    root, per_level = _run_lifecycle(
        resolution,
        clobber=clobber,
        manifest_name="prepare_manifest.json",
        work=lambda: execute_prepare(resolution.config),
    )
    manifest_path = root / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "dataset_hash": resolution.config.data.prepared.dataset_hash,
            "prepared_dataset_hash": resolution.config.data.prepared.prepared_dataset_hash,
            "prepared_dataset_id": resolution.config.data.prepared.prepared_dataset_id,
            "row_counts": {"slices_by_level": {str(k): v for k, v in per_level.items()}},
        }
    )
    atomic_write_json(manifest_path, manifest)
    table = Table(title=f"prepared {root}")
    table.add_column("level", justify="right")
    table.add_column("slices", justify="right")
    for level, count in per_level.items():
        table.add_row(str(level), str(count))
    console.print(table)


@app.command()
def train(
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Train stock Ultralytics over a SATI-prepared product."""
    from sati.train import canonical_last_checkpoint, execute_train

    if resume and resolved_config is None:
        raise typer.BadParameter("--resume requires --resolved-config")
    if resume and clobber:
        raise typer.BadParameter("--resume and --clobber are mutually exclusive")
    resolution = _resolve("train", overrides, config_dir, config, resolved_config)
    root = active_output_dir(resolution.config)
    if resume:
        lifecycle = RunLifecycle(root)
        lifecycle.resume(resolution.config)
        resume_from = canonical_last_checkpoint(resolution.config)
        try:
            checkpoint, digest = execute_train(
                resolution.config,
                resume_from=resume_from,
            )
        except BaseException as error:
            lifecycle.fail(error)
            raise
        lifecycle.complete(
            {
                "best_checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_hash": digest,
            }
        )
    else:
        checkpoint, digest = _run_lifecycle(
            resolution,
            clobber=clobber,
            work=lambda: execute_train(resolution.config),
            artifacts_from=lambda result, root: {
                "best_checkpoint": _rel(result[0], root),
                "checkpoint_hash": result[1],
            },
        )
    console.print(f"checkpoint: {checkpoint}")
    console.print(f"checkpoint_hash: {digest}")


def _checkpoint_source_run(checkpoint: Path) -> Path | None:
    if checkpoint.parent.name == "checkpoints":
        candidate = checkpoint.parent.parent
        if (candidate / "config" / "resolved.yaml").is_file():
            return candidate
    return None


@app.command()
def infer(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    input_path: Annotated[Path, typer.Option("--input", exists=True)],
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
) -> None:
    """Run sliced inference over explicit ad-hoc imagery."""
    from sati.infer import execute_infer

    resolution = _resolve("infer", overrides, config_dir, config, resolved_config)
    contract = apply_inference_contract(resolution.config, checkpoint)
    class_mapping = {int(key): value for key, value in contract.class_mapping.items()}
    results = _run_lifecycle(
        resolution,
        clobber=clobber,
        work=lambda: execute_infer(
            resolution.config,
            checkpoint=checkpoint,
            input_path=input_path,
            class_mapping=class_mapping,
        ),
        artifacts_from=lambda result, root: {"results": _rel(result, root)},
    )
    console.print(f"results: {results}")


@app.command(name="eval")
def eval_cmd(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
) -> None:
    """Run full-frame validation evaluation from a canonical checkpoint."""
    from sati.evaluation import execute_eval

    resolution = _resolve("eval", overrides, config_dir, config, resolved_config)
    apply_inference_contract(resolution.config, checkpoint)
    results, metrics = _run_lifecycle(
        resolution,
        clobber=clobber,
        work=lambda: execute_eval(resolution.config, checkpoint=checkpoint),
        artifacts_from=lambda result, root: {
            "results": _rel(result[0], root),
            "metrics": _rel(resolution.config.eval_outputs.metrics.dir / "metrics.json", root),
        },
    )
    console.print(f"results: {results}")
    table = Table(title="eval")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in ("mAP50_95", "mAP50", "mAP75", "n_frames", "n_final_merge"):
        value = metrics[key]
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


@app.command()
def export(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
    fmt: Annotated[str, typer.Option("--format", help="Ultralytics export format")] = "onnx",
) -> None:
    """Export a canonical checkpoint and portable inference config."""
    from sati.export import execute_export
    from sati.hashing import model_hash

    resolution = _resolve(
        "export",
        overrides,
        config_dir,
        config,
        resolved_config,
        source_run_dir=_checkpoint_source_run(checkpoint),
    )
    contract = apply_inference_contract(resolution.config, checkpoint)
    artifact, sidecar = _run_lifecycle(
        resolution,
        clobber=clobber,
        work=lambda: execute_export(
            resolution.config,
            checkpoint=checkpoint,
            fmt=fmt,
            contract=contract,
        ),
        manifest_name="export_manifest.json",
        artifacts_from=lambda result, root: {
            "model": _rel(result[0], root),
            "model_hash": model_hash(result[0]),
            "inference_config": _rel(result[1], root),
        },
    )
    console.print(f"artifact: {artifact}")
    console.print(f"inference config: {sidecar}")


@app.command()
def convert(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", help="conversion output directory"),
    ] = None,
    overrides: Overrides = None,
    config_dir: ConfigDirs = None,
    config: ConfigOpt = None,
    resolved_config: ResolvedOpt = None,
    clobber: ClobberOpt = False,
    to: Annotated[str, typer.Option("--to", help="manifests | coco | yolo")] = "manifests",
) -> None:
    """Convert COCO/YOLO/manifests without slicing."""
    from sati.convert import execute_convert

    if resolved_config is not None and output is not None:
        raise typer.BadParameter(
            "--output is an authored-config convenience and cannot override "
            "--resolved-config"
        )
    if resolved_config is None and output is None:
        raise typer.BadParameter("--output is required unless --resolved-config is used")
    extra = [*(overrides or [])]
    if output is not None:
        extra.append(f"prepare_outputs.dir={output}")
    resolution = _resolve("convert", extra, config_dir, config, resolved_config)
    converted = _run_lifecycle(
        resolution,
        clobber=clobber,
        work=lambda: execute_convert(resolution.config, to=to),
    )
    console.print(f"converted: {converted}")


def _register_inspect() -> None:
    from sati.inspect import inspect_app

    app.add_typer(inspect_app, name="inspect", help="Read-only artifact introspection.")


_register_inspect()


def main() -> None:
    try:
        app()
    except SatiError as error:
        # Any user-facing SATI refusal (lifecycle, ingest, train-gate, config
        # composition) exits cleanly. `main` runs outside Click's runtime, so
        # `typer.Exit` would escape unhandled (traceback, exit 1); `SystemExit`
        # is what the interpreter turns into a clean process exit code.
        console.print(f"[red]error:[/red] {error}")
        raise SystemExit(2) from error
