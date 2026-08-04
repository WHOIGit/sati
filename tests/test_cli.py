"""CLI lexical and exact-replay integration contracts."""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

import sati.cli as cli
from sati.cli import app
from sati.config_loader import resolve_configuration
from sati.config_loader.compositor import ConfigCompositionError
from sati.errors import SatiError
from sati.io import IngestError
from sati.lifecycle import LifecycleError
from sati.train import TrainRefusedError


@pytest.mark.parametrize(
    "error",
    [
        LifecycleError("output directory is not empty"),
        IngestError("COCO source not found"),
        TrainRefusedError("no prepared dataset"),
        ConfigCompositionError("mutually exclusive roots"),
    ],
)
def test_sati_errors_exit_cleanly_with_code_2(monkeypatch, capsys, error):
    def boom() -> None:
        raise error

    monkeypatch.setattr(cli, "app", boom)
    # Every user-facing SATI refusal must become a clean exit code 2 with a
    # message — never a traceback (regression: `main` raised typer.Exit outside
    # Click, which propagated as exit 1, and only LifecycleError was caught).
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    assert str(error) in capsys.readouterr().out


def test_all_user_facing_errors_share_the_sati_base():
    # The shared base is what makes `main`'s single except cover every refusal.
    for cls in (LifecycleError, IngestError, TrainRefusedError, ConfigCompositionError):
        assert issubclass(cls, SatiError)


def test_run_lifecycle_records_primary_artifacts(tmp_path):
    from sati.cli import _rel, _run_lifecycle

    resolution = resolve_configuration(
        [
            "data.source.uri=unused.json",
            "model.num_classes=1",
            f"infer_outputs.dir={tmp_path}/run",
        ],
        command="infer",
        base_dir=tmp_path,
    )
    result_path = tmp_path / "run" / "results" / "predictions.parquet"
    _run_lifecycle(
        resolution,
        clobber=False,
        work=lambda: result_path,
        artifacts_from=lambda result, root: {"results": _rel(result, root)},
    )
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    # Regression: the manifest used to complete with an empty artifacts map.
    assert manifest["artifacts"] == {"results": "results/predictions.parquet"}


def test_unknown_dash_prefixed_option_is_rejected():
    result = CliRunner().invoke(app, ["prepare", "--not-a-sati-option"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_convert_exact_resolved_replay_needs_no_output_override(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded by manifest conversion")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": image.name,
                        "width": 20,
                        "height": 10,
                    }
                ],
                "annotations": [],
                "categories": [{"id": 1, "name": "object"}],
            }
        )
    )
    output = tmp_path / "converted"
    resolution = resolve_configuration(
        [
            f"data.source.uri={annotations}",
            "model.num_classes=1",
            f"prepare_outputs.dir={output}",
        ],
        command="convert",
        base_dir=tmp_path,
    )
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(
        yaml.safe_dump(
            resolution.config.model_dump(mode="json", exclude_none=False),
            sort_keys=False,
        )
    )
    result = CliRunner().invoke(
        app,
        ["convert", "--resolved-config", str(resolved), "--to", "manifests"],
    )
    assert result.exit_code == 0, result.output
    assert (output / "config" / "resolved.yaml").is_file()
    assert (output / "manifests" / "fullframes.parquet").is_file()
