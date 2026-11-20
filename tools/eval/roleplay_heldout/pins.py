"""Pinned inputs and selection parameters of the held-out roleplay corpus.

Every value here is part of the corpus definition: changing the dataset revision, a file hash, the
seed, the per-stratum count, the filters in ``cleaning.py`` or the selection rule in ``corpus.py``
changes the corpus text and therefore its published SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET = "lemonilia/roleplaying-forums-raw"
REVISION = "6ba79e71d8266428eee590f634bf77aedead39b9"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
ENDPOINT = "https://huggingface.co"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "Forum scrapes collected and published by lemonilia "
    f"({DATASET_URL}, revision {REVISION}, licence {LICENSE}). The posts are the forum members' writing."
)

CORPUS_ID = "ninfer-rp-forum-fresh-v1"
DOMAIN = "roleplay_forum"
FILTER_VERSION = "rp-forum-filters/1"
SELECTION_VERSION = "rp-fresh-selection/1"
SEED = "20261125"
PER_STRATUM = 6

# SHA-256 of the corpus text of the default selection (see ``corpus.corpus_digest``); ``check`` compares
# a build against it.
PUBLISHED_CORPUS_SHA256: str | None = "61ea33b9e0ac6901078e035b044efc4d8f68ae783d73ae2a7414e84e8d465cc5"


@dataclass(frozen=True)
class Source:
    """One raw parquet file (one row per thread: ``thread_url``, ``thread_title``, ``contents``)."""

    stratum: str
    filename: str
    kind: str  # forum markup: "elliquiy" (print pages) or "xenforo"
    size: int | None = None
    sha256: str | None = None


SOURCES = (
    Source(
        "elliquiy-nonadult",
        "elliquiy-nonadult-rp-original--2023-03--original.parquet",
        "elliquiy",
        12265108,
        "23b6ca6dfe0cef642d7494a79f75ed1e749b6a871aeaa9d1e54005ca319669b7",
    ),
    Source(
        "elliquiy-oneshots",
        "elliquiy-one-shots-one-shot-stories-original--2023-03--original.parquet",
        "elliquiy",
        56665691,
        "f2bc0c128b24a7f54950c6b2736fb146ade0592af74cd614bf3b2b19cf59096d",
    ),
    Source(
        "bluemoon-group",
        "bluemoon-group-roleplays--2023-03--original.parquet",
        "xenforo",
        19059302,
        "3d12223d2f70600c60d63161d8f6bbcea7883dbd3c1391fdde3019b6fa4ca574",
    ),
    Source(
        "bluemoon-fanbased-1x1",
        "bluemoon-fanbased-roleplays-1x1--2023-03--original.parquet",
        "xenforo",
        128188336,
        "8ac2465334c1ae846cfb704435b5854ab1ed77c4685d7b03d493c2d20974c2ea",
    ),
    Source(
        "rpnation-quests",
        "RPNation--roleplays--quests.parquet",
        "xenforo",
        8766436,
        "a8fcf34fc6d4c2c2389afc0a9b56e7750e9c6c0f685a557b63d8f1556f7770c4",
    ),
    Source(
        "rpnation-1x1",
        "RPNation--roleplays--one-on-one--part-1.parquet",
        "xenforo",
        84288490,
        "fc8c0c3d60952138c93280cbcab5cc5ac9092fc371fb71feb56e8583defe92e9",
    ),
)
