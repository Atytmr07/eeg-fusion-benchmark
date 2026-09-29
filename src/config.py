"""Experiment configuration and content-addressed output directories.

Every Bonn run writes to a directory named after a hash of its full configuration,
so results produced with different settings can never overwrite each other.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results_v2"

# Bonn corpus (Andrzejak et al. 2001): 173.61 Hz, 4097 samples (23.6 s) per segment,
# already band-limited to 0.53-40 Hz at acquisition. Sets A and B are scalp EEG from
# healthy volunteers; C, D and E are intracranial recordings from patients.
FS = 173.61
SET_NAMES = ("A", "B", "C", "D", "E")

# Task definitions: set letter -> class index.
#   T1: three classes; the hard part is normal vs interictal.
#   T2: intracranial sets only, which removes the scalp vs intracranial confound.
#   T3: the binary split most used in the literature. Kept as a reference only: a
#       single log-variance feature already reaches 0.954 macro F1 on it.
TASKS = {
    "T1_3class": {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2},
    "T2_intracranial_binary": {"C": 0, "D": 0, "E": 1},
    "T3_classic_binary": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 1},
}

# Normalisation ablation (which signal each branch sees).
#   sig_z:       the raw branch sees the z-scored signal
#   spec_from_z: the spectrogram is computed from the z-scored signal
#   spec_z:      the spectrogram is additionally z-scored
# N2 is the main configuration. N1 is what the original notebook implementation ran
# by accident: a caching bug fed the spectrogram branch the unnormalised signal.
NORM_MODES = {
    "N0_raw_raw":       {"sig_z": False, "spec_from_z": False, "spec_z": False},
    "N1_z_rawspec":     {"sig_z": True,  "spec_from_z": False, "spec_z": False},
    "N2_z_zspec":       {"sig_z": True,  "spec_from_z": True,  "spec_z": False},
    "N3_z_zspec_specz": {"sig_z": True,  "spec_from_z": True,  "spec_z": True},
}

MODELS = ("raw1d", "spec2d", "early", "late", "gated", "attention", "score")


@dataclass(frozen=True)
class Config:
    task: str = "T1_3class"
    norm_mode: str = "N2_z_zspec"

    # Bonn is already band-limited at acquisition, so no extra filter by default.
    apply_lowpass: bool = False
    lowpass_hz: float = 40.0
    lowpass_order: int = 4

    # STFT for the spectrogram branch
    nfft: int = 256
    win: int = 256
    hop: int = 128
    fmax: float = 40.0

    # Training
    embed_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 60
    # Without a minimum, early stopping could fire while a model still predicted a
    # single class: validation F1 stayed flat, patience ran out around epoch 11, and
    # the collapsed weights were kept as "best" (runs with AUC 0.96 but macro F1 0.40).
    min_epochs: int = 20
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_weighted_loss: bool = True

    # Evaluation: n_repeats x n_folds stratified CV, with a stratified validation
    # split carved out of each training fold for early stopping and score fusion.
    n_folds: int = 5
    n_repeats: int = 5
    val_ratio: float = 0.2
    base_seed: int = 20260727

    # CPU thread count. It is part of the run identity because it changes results:
    # floating point summation order depends on it, which moves per-fold macro F1 by
    # up to 0.09 (see docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md).
    threads: int = 2

    models: tuple = field(default=MODELS)

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def outdir(self) -> Path:
        """Output directory of this run. Has no side effects (does not create it)."""
        return RESULTS_ROOT / f"{self.task}__{self.norm_mode}__{self.hash()}"

    def save(self, outdir: Path | None = None) -> Path:
        d = Path(outdir) if outdir else self.outdir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / "config.json"
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        (d / "env.json").write_text(json.dumps(runtime_env(), indent=2), encoding="utf-8")
        return p


def runtime_env() -> dict:
    """Environment facts that can affect results; saved next to every run.

    The effective thread count is recorded separately from Config.threads, because
    the two can differ if OMP_NUM_THREADS was not set before the process started.
    """
    import platform
    import sys

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    for mod in ("torch", "numpy", "scipy", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    try:
        import torch
        env["torch_num_threads_effective"] = torch.get_num_threads()
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return env


def fold_seed(base_seed: int, repeat: int, fold: int, model: str) -> int:
    """Deterministic, independent seed for each (repeat, fold, model) triple.

    Deriving the seed from the triple, rather than from a global RNG stream, makes
    every model's result independent of the order in which models are trained.
    """
    key = f"{base_seed}|{repeat}|{fold}|{model}".encode()
    return int(hashlib.sha1(key).hexdigest()[:8], 16) % (2**31 - 1)
