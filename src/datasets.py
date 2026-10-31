"""Which corpus builder and fold count belong to a dataset name.

Both corpora deliver the same form (18 bipolar channels, 256 Hz, 10 s windows, the same
labelling and subsampling), so every driver is shared; only the corpus differs.
Results go to results_v2/<dataset>/.
"""
from __future__ import annotations

DATASETS = ("chbmit", "siena")
N_FOLDS = {"chbmit": 23, "siena": 14}      # person-wise LOSO folds


def corpus_builder(name: str):
    """The build_corpus function of a dataset (same signature and output for both)."""
    if name == "chbmit":
        from .chbmit_corpus import build_corpus
    elif name == "siena":
        from .siena_corpus import build_corpus
    else:
        raise ValueError(f"unknown dataset: {name}")
    return build_corpus
