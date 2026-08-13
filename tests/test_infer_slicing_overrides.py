"""Infer-time slice-geometry overrides and the two preprocessing identities.

The defect these cover: ``apply_inference_contract`` used to replace ``cfg.slicing``
wholesale, so a resolved ``slicing.level_policy`` override was discarded before a
single slice was cut — silently, because the ``preprocessing_hash`` assertion
re-hashed the values it had just assigned and could never fail.
"""

from __future__ import annotations

import pytest

from sati.checkpoints import (
    GEOMETRY_SLICING_FIELDS,
    PINNED_SLICING_FIELDS,
    apply_inference_contract,
    build_inference_contract,
    embed_inference_contract,
    finalize_checkpoint,
)
from sati.config_loader import resolve_configuration
from sati.errors import PreprocessingOverrideRefused
from sati.hashing import preprocessing_hash
from sati.infer import PyramidDivergenceWarner, _level_scales, run_hashes_for
from sati.tiling import plan_frame

#: The pyramid the checkpoint under test is trained on. Overrides below move off it.
TRAINED_START_SCALE = 0.25
TRAINED_FACTOR = 3.0

#: An authored experiment root, because the level policy is a tagged union: the
#: packaged default is ``kind: auto``, which has no ``start_scale`` key, so an
#: inline ``slicing.level_policy.start_scale=`` against it is an append rather than
#: an override and Hydra's struct mode rejects it. Authoring the policy first is
#: also what the real invocation does — an experiment YAML plus inline overrides.
_ARM_TEMPLATE = """\
# @package _global_
defaults:
  - /base: default
  - /data: coco
  - /model: yolo26n
  - /slicing: default
  - /recombination: default
  - /prediction: default
  - /training: default
  - _self_

experiment:
  name: test/slicing-overrides

model:
  num_classes: 2

slicing:
  level_policy:
    kind: start_then_geometric
    start_scale: {start_scale}
    subsequent_downsample_factor: {factor}
    stop_when_long_side_lte_nominal: true
"""


def _arm_config(tmp_path, *, start_scale=TRAINED_START_SCALE, factor=TRAINED_FACTOR):
    path = tmp_path / f"arm_{start_scale}_{factor}.yaml"
    path.write_text(_ARM_TEMPLATE.format(start_scale=start_scale, factor=factor))
    return path


def _resolution(tmp_path, overrides=(), *, command="infer", config_file=None):
    return resolve_configuration(
        [
            "data.source.uri=unused.json",
            f"data.prepared.uri={tmp_path}/prepared",
            f"training_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command=command,
        config_file=config_file or _arm_config(tmp_path),
        base_dir=tmp_path,
    )


def _re_sliced(tmp_path, *, command="infer"):
    """Resolved config asking for a pyramid the checkpoint was not trained on."""
    return _resolution(
        tmp_path,
        [
            "slicing.level_policy.start_scale=0.5",
            "slicing.level_policy.subsequent_downsample_factor=2.0",
        ],
        command=command,
    )


def _checkpoint(tmp_path, torch):
    """A canonical checkpoint carrying a contract for the trained pyramid."""
    cfg = _resolution(tmp_path, command="train").config
    source = tmp_path / "native.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, source)
    artifact, _ = finalize_checkpoint(
        source,
        role="best",
        output_dir=tmp_path / "checkpoints",
        contract=build_inference_contract(cfg, class_mapping={0: "ring", 1: "spiral"}),
    )
    return artifact


def test_pinned_and_geometry_fields_partition_the_preprocessing_source():
    """Every field preprocessing_hash reads is either pinned or overridable.

    A field in neither bucket would be one the contract can silently restore.
    """
    from sati.config_schemas import SlicingConfig
    from sati.hashing import preprocessing_hash_full_source

    source = preprocessing_hash_full_source(SlicingConfig())
    slicing_fields = set(source) - {"version", "stretch_policy", "level_stack"}
    assert slicing_fields == set(PINNED_SLICING_FIELDS) | set(GEOMETRY_SLICING_FIELDS)
    assert not set(PINNED_SLICING_FIELDS) & set(GEOMETRY_SLICING_FIELDS)


def test_geometry_override_survives_the_contract(tmp_path):
    """The regression test for the silent discard."""
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _re_sliced(tmp_path).config
    apply_inference_contract(cfg, checkpoint)

    assert cfg.slicing.level_policy.start_scale == 0.5
    assert cfg.slicing.level_policy.subsequent_downsample_factor == 2.0
    # and it reaches the planner, not just the config object
    trained = _resolution(tmp_path).config
    assert len(plan_frame((8000, 6000), cfg.slicing).levels) != len(
        plan_frame((8000, 6000), trained.slicing).levels
    )


def test_effective_hash_moves_and_trained_hash_does_not(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _re_sliced(tmp_path).config
    contract = apply_inference_contract(cfg, checkpoint)
    hashes = run_hashes_for(
        cfg,
        checkpoint=checkpoint,
        trained_preprocessing_hash=contract.hashes["preprocessing_hash"],
    )
    assert hashes.preprocessing_hash == contract.hashes["preprocessing_hash"]
    assert hashes.effective_preprocessing_hash == preprocessing_hash(cfg.slicing)
    assert hashes.effective_preprocessing_hash != hashes.preprocessing_hash


def test_unoverridden_run_has_equal_preprocessing_identities(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(tmp_path).config
    contract = apply_inference_contract(cfg, checkpoint)
    hashes = run_hashes_for(
        cfg,
        checkpoint=checkpoint,
        trained_preprocessing_hash=contract.hashes["preprocessing_hash"],
    )
    assert hashes.effective_preprocessing_hash == hashes.preprocessing_hash


@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("nominal_slice_size", "slicing.nominal_slice_size=512"),
        ("image_format", "slicing.image_format=jpeg"),
    ],
)
@pytest.mark.parametrize("command", ["infer", "eval", "export"])
def test_pinned_field_divergence_refuses(tmp_path, field_name, override, command):
    """Model-coupled fields hard fail — they never warn and never restore."""
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(tmp_path, [override], command=command).config
    with pytest.raises(PreprocessingOverrideRefused) as excinfo:
        apply_inference_contract(cfg, checkpoint, pin_geometry=(command == "export"))
    assert excinfo.value.field_name == field_name


def test_inconsistent_contract_fails_verification(tmp_path):
    """The check used to re-hash values it had just assigned, so it never fired."""
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(tmp_path).config

    contract = build_inference_contract(cfg, class_mapping={0: "ring", 1: "spiral"})
    tampered = contract.model_copy(deep=True)
    tampered.slicing["overlap_ratio"] = 0.42  # digest no longer describes the fields
    embed_inference_contract(checkpoint, tampered)

    with pytest.raises(ValueError, match="preprocessing_hash failed verification"):
        apply_inference_contract(_resolution(tmp_path).config, checkpoint)


def test_export_pins_geometry_and_warns(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _re_sliced(tmp_path, command="export").config

    with pytest.warns(UserWarning, match="ignoring slice-geometry override"):
        contract = apply_inference_contract(cfg, checkpoint, pin_geometry=True)

    # the exported config describes the trained pyramid, not the requested one
    assert cfg.slicing.level_policy.start_scale == TRAINED_START_SCALE
    assert preprocessing_hash(cfg.slicing) == contract.hashes["preprocessing_hash"]


def test_export_without_geometry_override_is_silent(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(tmp_path, command="export").config
    with warnings_as_errors():
        apply_inference_contract(cfg, checkpoint, pin_geometry=True)


def warnings_as_errors():
    import warnings as _w

    context = _w.catch_warnings()
    context.__enter__()
    _w.simplefilter("error")
    return _WarningGuard(context)


class _WarningGuard:
    def __init__(self, context):
        self._context = context

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._context.__exit__(*exc_info)
        return False


def _warner(tmp_path, torch, *, start_scale, factor):
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(
        tmp_path,
        [
            f"slicing.level_policy.start_scale={start_scale}",
            f"slicing.level_policy.subsequent_downsample_factor={factor}",
        ],
    ).config
    contract = apply_inference_contract(cfg, checkpoint)
    return PyramidDivergenceWarner(cfg, contract), cfg


def test_divergence_warns_when_stacks_share_no_scale(tmp_path):
    torch = pytest.importorskip("torch")
    # trained stack starts at 0.25 and thirds; this one starts at 0.9 and halves,
    # so no planned level scale coincides
    warner, cfg = _warner(
        tmp_path, torch, start_scale=0.9, factor=2.0
    )
    plan = plan_frame((8000, 6000), cfg.slicing)
    with pytest.warns(UserWarning, match="differs from training stack"):
        warner.check(plan)


def test_divergence_warns_when_stacks_partially_overlap(tmp_path):
    torch = pytest.importorskip("torch")
    # keeps the trained 0.25 start, only deepens the stack, so the two share
    # scales — partial overlap is still divergence and must not be silenced
    warner, cfg = _warner(
        tmp_path, torch, start_scale=0.25, factor=2.0
    )
    plan = plan_frame((8000, 6000), cfg.slicing)
    trained_scales = _level_scales(
        plan_frame((8000, 6000), _resolution(tmp_path).config.slicing)
    )
    runtime_scales = _level_scales(plan)
    assert set(runtime_scales) & set(trained_scales)  # genuinely overlapping
    assert runtime_scales != trained_scales
    with pytest.warns(UserWarning, match="differs from training stack"):
        warner.check(plan)


def test_divergence_silent_only_when_stacks_are_identical(tmp_path):
    """A policy change that plans the same stack is not a divergence.

    This is why the comparison is on planned scales rather than policy fields.
    """
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(tmp_path, torch)
    cfg = _resolution(tmp_path).config  # trained policy, unchanged
    contract = apply_inference_contract(cfg, checkpoint)
    warner = PyramidDivergenceWarner(cfg, contract)
    with warnings_as_errors():
        warner.check(plan_frame((8000, 6000), cfg.slicing))


def test_divergence_warns_once_per_command(tmp_path):
    torch = pytest.importorskip("torch")
    warner, cfg = _warner(
        tmp_path, torch, start_scale=0.9, factor=2.0
    )
    plan = plan_frame((8000, 6000), cfg.slicing)
    with pytest.warns(UserWarning):
        warner.check(plan)
    with warnings_as_errors():  # eval plans a stack per frame; only the first warns
        warner.check(plan)
        warner.check(plan)


def test_divergence_never_refuses(tmp_path):
    torch = pytest.importorskip("torch")
    warner, cfg = _warner(
        tmp_path, torch, start_scale=0.99, factor=2.0
    )
    plan = plan_frame((8000, 6000), cfg.slicing)
    with pytest.warns(UserWarning):
        warner.check(plan)  # returns normally — advisory, never fatal
