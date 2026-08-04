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
