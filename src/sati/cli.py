"""Typer entrypoints (v1 §13). Thin: parse args, compose, hand the typed
``RootConfig`` to a CLI-free work function. No logic here.

The dash rule: ``--dashed`` args are Typer options; dash-free ``a.b=c`` args
are Hydra overrides forwarded to the compositor (``experiment=name`` selects
an experiment root). Commands land here as their work functions are built —
absent commands are unsupported, not stubbed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from sati.config_loader import compose_and_resolve

app = typer.Typer(name="sati", no_args_is_help=True, add_completion=False)
console = Console()

Overrides = Annotated[Optional[list[str]], typer.Argument(help="Hydra overrides (a.b=c)")]
ConfigDirOpt = Annotated[
    Optional[Path], typer.Option("--config-dir", help="extra config dir (highest precedence)")
]
ResolvedOpt = Annotated[
    Optional[Path], typer.Option("--resolved-config", help="re-run a resolved.yaml verbatim")
]


def _compose(overrides, config_dir, resolved_config, *, write=True):
    return compose_and_resolve(
        overrides or [],
        config_dirs=[config_dir] if config_dir else (),
        resolved_config=resolved_config,
        write=write,
    )


@app.command()
def prepare(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
) -> None:
    """Ingest COCO/YOLO, slice all levels, write manifests + the YOLO tree."""
    from sati.materialize import execute_prepare

    cfg = _compose(overrides, config_dir, resolved_config)
    root, per_level = execute_prepare(cfg)
    table = Table(title=f"prepared {root}")
    table.add_column("level", justify="right")
    table.add_column("slices", justify="right")
    for level, count in per_level.items():
        table.add_row(str(level), str(count))
    console.print(table)


@app.command()
def train(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
) -> None:
    """Stock ultralytics over the materialized YOLO tree."""
    from sati.train import execute_train

    cfg = _compose(overrides, config_dir, resolved_config)
    checkpoint, ckpt_hash = execute_train(cfg)
    console.print(f"checkpoint: {checkpoint}")
    console.print(f"checkpoint_hash: {ckpt_hash}")


@app.command()
def infer(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
) -> None:
    """Sliced inference on new imagery; results to validation_results.parquet."""
    from sati.infer import execute_infer

    cfg = _compose(overrides, config_dir, resolved_config)
    results = execute_infer(cfg)
    console.print(f"results: {results}")


@app.command(name="eval")
def eval_cmd(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
) -> None:
    """Sliced inference over the prepared validation split + mAP scoring."""
    from sati.evaluation import execute_eval

    cfg = _compose(overrides, config_dir, resolved_config)
    results, metrics = execute_eval(cfg)
    console.print(f"results: {results}")
    table = Table(title="eval")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in ("mAP50_95", "mAP50", "mAP75", "n_frames", "n_final_merge"):
        table.add_row(key, f"{metrics[key]:.4f}" if isinstance(metrics[key], float) else str(metrics[key]))
    for name, ap in metrics["per_class_ap50_95"].items():
        table.add_row(f"AP[{name}]", f"{ap:.4f}")
    console.print(table)


@app.command()
def export(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
    fmt: Annotated[str, typer.Option("--format", help="ultralytics export format")] = "onnx",
) -> None:
    """Export a checkpoint + write the inference_config.yaml sidecar."""
    from sati.export import execute_export

    cfg = _compose(overrides, config_dir, resolved_config)
    artifact, sidecar = execute_export(cfg, fmt=fmt)
    console.print(f"artifact: {artifact}")
    console.print(f"inference config: {sidecar}")


@app.command()
def convert(
    overrides: Overrides = None,
    config_dir: ConfigDirOpt = None,
    resolved_config: ResolvedOpt = None,
    to: Annotated[str, typer.Option("--to", help="target: manifests | coco | yolo")] = "manifests",
) -> None:
    """Format interchange between COCO/YOLO and canonical manifests. No slicing."""
    from sati.convert import execute_convert

    cfg = _compose(overrides, config_dir, resolved_config)
    out = execute_convert(cfg, to=to)
    console.print(f"converted → {out}")


def _register_inspect() -> None:
    from sati.inspect import inspect_app

    app.add_typer(inspect_app, name="inspect", help="Read-only artifact introspection.")


_register_inspect()


def main() -> None:
    app()
