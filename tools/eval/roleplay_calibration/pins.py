"""Selection parameters of the roleplay calibration text.

The dataset, its pinned files, the cleaning rules and the minors screen are those of
``tools.eval.roleplay_heldout.pins`` and ``tools.eval.roleplay_heldout.cleaning`` and are not repeated here. Every
value below is part of the corpus definition: changing the seed, a quota, the reserve, an exclusion list or the
filters changes the text and therefore its SHA-256.
"""

from __future__ import annotations

CORPUS_ID = "ninfer-rp-forum-calibration-v1"
SELECTION_VERSION = "rp-calibration-selection/1"
SEED = "20261130"

# Threads (one excerpt each) per raw file. Equal weights across the strata the evaluation sets draw from, except the
# smallest file (388 threads, about one in ten survives the thread rules), which cannot supply more.
QUOTAS = {
    "elliquiy-nonadult": 72,
    "elliquiy-oneshots": 72,
    "bluemoon-group": 72,
    "bluemoon-fanbased-1x1": 72,
    "rpnation-quests": 24,
    "rpnation-1x1": 72,
}

# Extra surviving threads drawn per stratum, in hash order, so a thread dropped by the shingle check against the
# evaluation texts is replaced by the next one in the same order.
RESERVE = 12

# SHA-256 of the corpus text of the default selection (``corpus.corpus_digest``); ``check`` compares a build against it.
PUBLISHED_CORPUS_SHA256: str | None = "2232e11f81f74c2f5284f576c8ed553501f5a7b03976dc46dd8d91a6c63506a2"
