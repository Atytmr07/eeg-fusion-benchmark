"""CHB-MIT Scalp EEG: EDF reading, seizure annotation parsing, windowing.

Why our own EDF reader: EDF (Kemp et al. 1992) is a fixed-width ASCII header followed
by int16 data records, and reading it takes under a hundred lines. Keeping the reader
here, rather than depending on mne or pyedflib, keeps the dependency surface small
and makes visible how the file is interpreted. Scaling and channel selection affect
the results directly, so we prefer them visible.

Source:   https://physionet.org/content/chbmit/1.0.0/
License:  Open Data Commons Attribution License v1.0
Cite:     Shoeb (2009), MIT PhD thesis; Goldberger et al. (2000), Circulation 101(23).

Differences from Bonn that shape the design:
  - 23 channels (24 or 26 in some recordings) instead of one
  - 256 Hz instead of 173.61 Hz
  - continuous recordings instead of pre-cut segments
  - extreme class imbalance: seizures are under 1 percent of the recording time
  - subject identifiers exist, so subject-wise cross-validation is possible. That is
    the main reason to use this corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_ROOT

CHB_ROOT = DATA_ROOT / "chbmit"
FS_CHB = 256.0

# Standard 18-channel longitudinal bipolar montage.
#
# Why a fixed list rather than the intersection of all recordings: in CHB-MIT the
# channel set varies not only between subjects but between recordings of the same
# subject. Part of chb12 was recorded with a CS2-referenced montage (C3-CS2, CP2-CS2,
# ...) that shares not a single channel with the standard one, so the intersection
# over all recordings is empty; a naive intersection silently yields zero channels.
#
# With this fixed target 683 of 686 recordings and 185 of 198 seizures are kept, and
# every subject remains usable. All 3 dropped recordings and 13 dropped seizures come
# from the differently recorded part of chb12.
TARGET_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)


# --- EDF reading ----------------------------------------------------------------------

@dataclass
class EdfHeader:
    n_records: int
    record_duration: float        # seconds
    labels: list[str]
    samples_per_record: list[int]
    phys_min: np.ndarray
    phys_max: np.ndarray
    dig_min: np.ndarray
    dig_max: np.ndarray
    header_bytes: int

    @property
    def fs(self) -> list[float]:
        return [s / self.record_duration for s in self.samples_per_record]

    @property
    def duration(self) -> float:
        return self.n_records * self.record_duration


def read_edf_header(path: Path) -> EdfHeader:
    """Read the fixed EDF header: 256 bytes general + 256 bytes per signal."""
    with open(path, "rb") as fh:
        raw = fh.read(256)
        if len(raw) < 256:
            raise ValueError(f"{path.name}: header too short, file may be corrupt")
        header_bytes = int(raw[184:192].decode("ascii", "replace").strip() or 0)
        n_records = int(raw[236:244].decode("ascii", "replace").strip())
        record_duration = float(raw[244:252].decode("ascii", "replace").strip())
        ns = int(raw[252:256].decode("ascii", "replace").strip())

        def field(width: int) -> list[str]:
            buf = fh.read(width * ns).decode("ascii", "replace")
            return [buf[i * width:(i + 1) * width].strip() for i in range(ns)]

        labels = field(16)
        field(80)                                   # transducer type (unused)
        field(8)                                    # physical dimension (unused)
        phys_min = np.array([float(x) for x in field(8)])
        phys_max = np.array([float(x) for x in field(8)])
        dig_min = np.array([float(x) for x in field(8)])
        dig_max = np.array([float(x) for x in field(8)])
        field(80)                                   # prefiltering (unused)
        spr = [int(x) for x in field(8)]

    return EdfHeader(n_records, record_duration, labels, spr,
                     phys_min, phys_max, dig_min, dig_max,
                     header_bytes or (256 * (ns + 1)))


def read_edf(path: Path, channels: list[str] | None = None) -> tuple[np.ndarray, list[str], float]:
    """Signals in physical units (microvolts), shape (channel, sample).

    All channels must share one sampling rate; this holds for CHB-MIT and an error is
    raised otherwise.
    """
    h = read_edf_header(path)
    if len(set(h.samples_per_record)) != 1:
        raise ValueError(f"{path.name}: channels have different sampling rates, "
                         f"not supported by this reader")
    spr = h.samples_per_record[0]
    ns = len(h.labels)

    data = np.fromfile(path, dtype="<i2", offset=h.header_bytes)
    expected = h.n_records * ns * spr
    if data.size < expected:                        # truncated file: keep complete records
        h.n_records = data.size // (ns * spr)
        expected = h.n_records * ns * spr
    # Records are stored as [record][signal][sample]; reorder to [signal][time].
    data = data[:expected].reshape(h.n_records, ns, spr)
    data = data.transpose(1, 0, 2).reshape(ns, -1).astype(np.float32)

    # EDF scaling: physical = (digital - dig_min) * gain + phys_min
    span = np.where(h.dig_max - h.dig_min == 0, 1.0, h.dig_max - h.dig_min)
    gain = ((h.phys_max - h.phys_min) / span).astype(np.float32)
    offs = h.phys_min.astype(np.float32)
    dmin = h.dig_min.astype(np.float32)
    data = (data - dmin[:, None]) * gain[:, None] + offs[:, None]

    labels = h.labels
    if channels is not None:
        idx = [labels.index(c) for c in channels]
        data, labels = data[idx], list(channels)
    return data, labels, h.fs[0]


# --- Seizure annotations --------------------------------------------------------------

@dataclass
class Seizure:
    file: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def parse_summary(path: Path, strict: bool = False) -> dict[str, list[Seizure]]:
    """Seizure intervals per file, from a case's chbNN-summary.txt.

    Parsing is SEQUENCE-based and does not trust the printed seizure number, because
    of a labelling error in the source data. In the chb09_08.edf block the second
    seizure's end line is labelled "Seizure 1 End Time":

        Seizure 1 Start Time: 2951 seconds
        Seizure 1 End Time:   3030 seconds
        Seizure 2 Start Time: 9196 seconds
        Seizure 1 End Time:   9267 seconds     <- should read "Seizure 2"

    A parser that pairs lines by number would treat 2951-9267 as one seizure, i.e.
    invent a 6316-second seizure and label hundreds of seizure-free windows as ictal.
    Sequence-based reading is immune: every "Start Time" opens a seizure and every
    "End Time" closes the most recently opened one.

    The number of seizures parsed in each block is checked against its "Number of
    Seizures in File". With strict=True a mismatch raises, otherwise it prints a
    warning.

    Annotations also exist as binary .edf.seizures files, but that format is not
    documented; the summary file is human-readable and the official source.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[Seizure]] = {}
    current: str | None = None
    declared: dict[str, int] = {}
    open_start: float | None = None

    def close_block() -> None:
        if current is None:
            return
        want, got = declared.get(current), len(out.get(current, []))
        if want is not None and want != got:
            msg = (f"{path.name}: {current} declares {want} seizures, "
                   f"{got} parsed")
            if strict:
                raise ValueError(msg)
            print(f"warning: {msg}")

    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            close_block()
            current = m.group(1)
            out.setdefault(current, [])
            open_start = None
            continue
        if current is None:
            continue
        m = re.match(r"Number of Seizures in File:\s*(\d+)", line)
        if m:
            declared[current] = int(m.group(1))
            continue
        m = re.match(r"Seizure.*Start Time:\s*(\d+)\s*seconds", line)
        if m:
            if open_start is not None:
                print(f"warning: {path.name}: unclosed seizure in {current} "
                      f"(start {open_start}) skipped")
            open_start = float(m.group(1))
            continue
        m = re.match(r"Seizure.*End Time:\s*(\d+)\s*seconds", line)
        if m:
            end = float(m.group(1))
            if open_start is None:
                print(f"warning: {path.name}: end line without a start in "
                      f"{current} ({end}) skipped")
                continue
            out[current].append(Seizure(current, open_start, end))
            open_start = None
    close_block()
    return out


def subject_files(subject: str, require_channels: bool = True) -> list[Path]:
    """EDF files of a case.

    The pattern is `{subject}*.edf`, not `{subject}_*.edf`: chb17's recordings are
    named chb17a_03.edf, chb17b_69.edf, chb17c_..., and a pattern expecting an
    underscore silently skips the whole case.

    With require_channels=True, recordings lacking any of TARGET_CHANNELS are dropped.
    """
    files = sorted((CHB_ROOT / subject).glob(f"{subject}*.edf"))
    if not require_channels:
        return files
    want = set(TARGET_CHANNELS)
    return [f for f in files if want <= set(read_edf_header(f).labels)]


# --- Windowing --------------------------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def seizure_ids(subject: str) -> dict[str, list[tuple[float, float, str]]]:
    """A persistent id for every seizure of a case, e.g. chb09_08.edf#2.

    The id lets windows be grouped by seizure. With overlapping windows, windows from
    the same seizure are near-identical; if some end up in training and some in test,
    the model's memorisation looks like generalisation. This error is documented in
    the CHB-MIT literature (Ali et al. 2024, doi:10.1098/rsos.230601).
    """
    summ = parse_summary(CHB_ROOT / subject / f"{subject}-summary.txt")
    out: dict[str, list[tuple[float, float, str]]] = {}
    for fname, zs in summ.items():
        for k, z in enumerate(zs, 1):
            out.setdefault(fname, []).append((z.start_s, z.end_s, f"{fname}#{k}"))
    return out


def plan_windows(path: Path, seizures: list[Seizure], win_s: float = 10.0,
                 stride_s: float | None = None,
                 guard_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Window labels computed WITHOUT reading the signal.

    A label depends only on the window's time and the seizure intervals, and the
    recording length comes from the EDF header. Deciding which windows to keep
    therefore needs no signal data, which is what makes the full corpus fit in memory.

    Returns y (0 non-ictal / 1 ictal) and t0 (window start, seconds).

    A window is ictal if it overlaps a seizure at all. Boundary windows (partial
    overlap) have an ambiguous label: with guard_s > 0 they are dropped instead. The
    main runs use guard_s = 0 (boundary windows kept, labelled ictal).
    """
    h = read_edf_header(path)
    fs = int(round(h.fs[0]))
    n = int(win_s * fs)
    step = int((stride_s if stride_s else win_s) * fs)
    total = int(h.duration * fs)
    sz = [(s.start_s, s.end_s) for s in seizures]

    y, t0 = [], []
    for start in range(0, total - n + 1, step):
        a0, a1 = start / fs, (start + n) / fs
        ov = max((_overlap(a0, a1, b0, b1) for b0, b1 in sz), default=0.0)
        if ov == 0.0:
            lab = 0
        elif ov >= win_s - 1e-9:                    # window entirely inside a seizure
            lab = 1
        else:                                       # boundary window
            if guard_s > 0:
                continue
            lab = 1
        y.append(lab)
        t0.append(a0)
    return np.array(y, np.int64), np.array(t0, np.float32)


def extract_windows(path: Path, channels: list[str], starts_s: np.ndarray,
                    win_s: float = 10.0) -> np.ndarray:
    """Read the windows starting at the given times. Returns (window, channel, sample)."""
    x, _, fs = read_edf(path, channels)
    fs = int(round(fs))
    n = int(win_s * fs)
    out = np.empty((len(starts_s), len(channels), n), np.float32)
    for i, t in enumerate(starts_s):
        s = int(round(float(t) * fs))
        out[i] = x[:, s:s + n]
    return out
