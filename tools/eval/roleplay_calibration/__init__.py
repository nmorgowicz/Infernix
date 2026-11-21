"""Roleplay-forum calibration text for importance matrices (``docs/roleplay-heldout.md``, "Calibration text").

``python -m tools.eval.roleplay_calibration build|check|mix``. Reuses the frozen cleaning, minors screen and
thread selection of ``tools.eval.roleplay_heldout`` on the same pinned raw files, with its own seed, exclusion
list and per-stratum counts. The text is never committed: it is rebuilt in a camp from the pinned dataset.
"""
