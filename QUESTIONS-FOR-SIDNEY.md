# Questions for Sidney

Non-blocking questions and provisional decisions, per standing instruction. Each item states
what was done in the meantime, so nothing here blocks work.

## 2026-07-02 — pyproject bootstrap

1. **`requires-python` floor.** Dev env is the repo `.venv` (Python 3.14); I set
   `requires-python = ">=3.12"` as a reasonable public floor, but only 3.14 is actually
   exercised. Happy to raise to `>=3.14` if SATI is effectively venv-only.
2. **opencv-python vs opencv-python-headless.** DESIGN-DOC-SYNTH specified `-headless`,
   but the two packages conflict in one env and SATI core already needs OpenCV. I put
   plain `opencv-python` in core deps and the synth extra reuses it. Fine unless you want
   headless-only CI images later.
3. **OpenCV 5.0.0 landed.** The venv resolved `opencv-python==5.0.0.93` (new major; docs
   were written assuming 4.x API). I'll verify `fillPoly`/`INTER_AREA`/`resize` behavior
   at implementation time and flag anything that shifted.
4. **`[ultralytics]` extra not installed yet** (pulls torch, multi-GB). I'll install it
   when `train`/`eval` work starts unless you want it now. Range pinned `>=8.3,<9` per the
   v1 §7 tripwire NOTE.
5. **Project metadata.** `pyproject.toml` has no `license`, `authors`, or repo URL fields
   yet — tell me what you want there whenever.
