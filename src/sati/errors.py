"""Shared base for user-facing SATI refusals.

A ``SatiError`` is a user-facing refusal: bad input, an unmet precondition, or
an invalid configuration — not an internal bug. The CLI (`sati.cli.main`)
catches this base and exits with code 2 and a plain message instead of a
traceback. Subclass it for any new user-facing refusal so the clean-exit
behavior applies automatically.

This module imports nothing from ``sati`` (or any heavy dependency) so the CLI
can reference the base without pulling image/backend modules into startup.
"""

from __future__ import annotations


class SatiError(RuntimeError):
    """User-facing refusal handled by the CLI as a clean exit."""


class PreprocessingOverrideRefused(SatiError):
    """A model-coupled preprocessing field disagrees with the checkpoint.

    Raised when the resolved ``slicing.nominal_slice_size`` or
    ``slicing.image_format`` differs from the value recorded in the checkpoint's
    inference contract. These two fields are pinned because the emitted slice
    size derives ``model.imgsz``: cutting tiles at one size while telling the
    backend another is a defect, not an experiment. The remaining slice geometry
    (level policy, overlap ratio, level cap) is overridable at infer/eval time.

    Refusal here is by value, not by override provenance — an authored config
    that disagrees with its checkpoint is as much an incompatibility as an inline
    override, and both stop the run.
    """

    def __init__(self, field_name: str, resolved: object, contract: object) -> None:
        super().__init__(
            f"slicing.{field_name} is pinned by the checkpoint's inference contract: "
            f"resolved {resolved!r}, contract {contract!r}. This field is model-coupled "
            f"and cannot be overridden at infer/eval/export time; override "
            f"slicing.level_policy, slicing.overlap_ratio, or slicing.max_levels instead, "
            f"or train a checkpoint at the slice size you want."
        )
        self.field_name = field_name
        self.resolved = resolved
        self.contract = contract
