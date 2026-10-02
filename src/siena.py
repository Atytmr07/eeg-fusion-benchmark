"""Siena Scalp EEG: seizure list parsing and conversion to the CHB-MIT format.

Siena is the third corpus: an external check of whether the CHB-MIT conclusions carry
over to a different hospital, montage, amplifier and patient population (adults instead
of children). Everything downstream (windowing, subsampling, splits, models) is shared
with CHB-MIT, so this module's only job is to deliver each recording in exactly the
CHB-MIT form: the 18 bipolar channels of TARGET_CHANNELS, at 256 Hz, in microvolts.

Source:   https://physionet.org/content/siena-scalp-eeg/1.0.0/
License:  Creative Commons Attribution 4.0 International
Cite:     Detti et al. (2020), Processes 8(7):846, doi:10.3390/pr8070846;
          Detti (2020), PhysioNet, doi:10.13026/5d4a-j060;
          Goldberger et al. (2000), Circulation 101(23).

Differences from CHB-MIT that shape the design:
  - referential 10-20 recordings (29 EEG electrodes, 20 in PN10) instead of the
    bipolar montage; the bipolar channels are derived by subtraction
  - 512 Hz instead of 256 Hz; resampled on the continuous recording
  - old electrode names (T3, T4, T5, T6 for T7, T8, P7, P8), label case varying
    between files (Fp2 / FP2, Cz / CZ)
  - seizure times given as wall-clock times (hh.mm.ss), not seconds from the start of
    the file, in hand-written lists with typos (see CORRECTIONS)
  - every one of the 41 recordings contains at least one seizure
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .chbmit import FS_CHB, TARGET_CHANNELS, Seizure, read_edf_header
from .config import DATA_ROOT

SIENA_ROOT = DATA_ROOT / "siena"
FS_SIENA = 512.0
SUBJECTS = ("PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09", "PN10", "PN11",
            "PN12", "PN13", "PN14", "PN16", "PN17")

# Modern 10-20 names used by TARGET_CHANNELS -> the older names used in Siena.
OLD_NAMES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def bipolar_pairs() -> list[tuple[str, str]]:
    """TARGET_CHANNELS as (anode, cathode) pairs in Siena's electrode names.

    FP1-F7 = Fp1 - F7, T7-P7 = T3 - T5, and so on. Names are upper case; EDF labels
    are matched case-insensitively (electrode_label).
    """
    out = []
    for ch in TARGET_CHANNELS:
        a, b = ch.split("-")
        out.append((OLD_NAMES.get(a, a), OLD_NAMES.get(b, b)))
    return out


ELECTRODES = tuple(sorted({e for pair in bipolar_pairs() for e in pair}))   # 19


def electrode_label(label: str) -> str:
    """'EEG Fp1' -> 'FP1'. Non-EEG labels ('EKG EKG', 'SPO2', '1') are returned as they
    are, upper-cased, and never match an electrode name."""
    label = label.strip()
    if label.upper().startswith("EEG "):
        label = label[4:]
    return label.strip().upper()


# --- Seizure lists ----------------------------------------------------------------------
#
# Every fix to the published Seizures-list-PNxx.txt files is listed here, with the
# reason. The parser is strict: a time it cannot read unambiguously raises instead of
# being guessed, so a new typo cannot slip through silently.

CORRECTIONS = {
    # Literal text replacements applied to a list before parsing.
    "text": {
        "PN10": [
            # "Registration start time:1 6.49.25": a space inside the hour. The next two
            # blocks of the same file give 16.49.25, and so does the EDF header.
            ("1 6.49.25", "16.49.25"),
        ],
    },
    # Misspelt file names -> the file in RECORDS.
    "file_names": {
        "PN01.edf": "PN01-1.edf",       # the only recording of PN01
        "PNO6-1.edf": "PN06-1.edf",     # letter O instead of digit 0
        "PNO6-2.edf": "PN06-2.edf",
        "PNO6-4.edf": "PN06-4.edf",
        "PN11-.edf": "PN11-1.edf",      # the only recording of PN11
    },
    # Registration start times that disagree with the EDF header, (subject, file) ->
    # header value. Seizure times are always converted with the header start time
    # (see recording_start), so these entries only document the disagreement; they are
    # verified against the headers by `python -m src.siena_download --audit`.
    "registration_start": {
        # List 16.17.45; header 19.17.45. The header is right: 19.17.45 plus the
        # header duration (41995 s) gives the listed end time 06.57.40 exactly, while
        # 16.17.45 would make the recording three hours longer than the file.
        ("PN14", "PN14-3.edf"): "19.17.45",
        # List 06.01.23; header 06.01.13. Not settled: in 18 recordings the listed end
        # is exactly 20 s before header start + duration (in 22 it equals it), and
        # the list's 06.01.23 fits that pattern (-20 s) while the header's 06.01.13
        # gives -10 s, which fits neither. The header is used, as for every
        # recording, because its start time belongs to the first sample by
        # definition; the 10 s matter for this 30 s seizure (open question).
        ("PN05", "PN05-3.edf"): "06.01.13",
    },
    # Seizures listed without their own registration times; they take them from the
    # previous block of the same file. Any other block without registration times is
    # an error.
    #   PN01: file name and registration times precede "Seizure n 1" and are shared by
    #         both seizures of the single recording, so seizure 2 has none of its own.
    #   PN12: seizure 2 repeats the file name of seizure 1 but no registration times.
    "inherited_registration": {("PN01", 2), ("PN12", 2)},
    # Every list names channel 5 "1"; the EDF headers label it "EEG O1". Only header
    # labels are used, so this needs no fix in the data path; it is recorded here
    # because it is the list's only channel-name error.
    "channel_labels": {"Channel 5: 1": "O1"},
}

# Ambiguities in the lists that the advisor decides. Both options are implemented;
# the active choice is recorded in every corpus' info.
DECISIONS = {
    # PN00 seizure 3 (PN00-3.edf): end "19.29.29", but the recording ends at 18.57.13,
    # which would make a 61-minute seizure running past the end of the file.
    #   "typo"     read it as 18.29.29, a 60 s seizure (in line with this patient's
    #              other four seizures, 54 to 74 s)
    #   "exclude"  drop the seizure and every window from its onset to the end of the
    #              recording, since the true end is unknown
    "PN00_seizure3": "typo",
    # PN10 seizure 3 (PN10-3.edf): "15.43.53 (CLINICAL ONSET); 15.43.59 (ELECTRIC
    # ONSET)".
    #   "electric"  15.43.59; CHB-MIT annotates electrographic onsets, so this keeps
    #               the two corpora consistent
    #   "clinical"  15.43.53
    # PN10 seizure 6 gives only a clinical onset ("15.18.26 (CLINICAL ONSET)"); it is
    # used as listed under either choice.
    "PN10_seizure3_onset": "electric",
    # PN10 seizure 2 (PN10-2.edf): end "11.41.04 opure 11.40.43" (Italian "oppure",
    # "or").
    #   "first"   11.41.04
    #   "second"  11.40.43
    "PN10_seizure2_end": "first",
}

_DECISION_OPTIONS = {
    "PN00_seizure3": ("typo", "exclude"),
    "PN10_seizure3_onset": ("electric", "clinical"),
    "PN10_seizure2_end": ("first", "second"),
}

_TIME = re.compile(r"\b(\d{2})[.:](\d{2})[.:](\d{2})\b")


def _seconds(token: str) -> int:
    h, m, s = (int(v) for v in _TIME.fullmatch(token).groups())
    if not (h < 24 and m < 60 and s < 60):
        raise ValueError(f"not a clock time: {token}")
    return 3600 * h + 60 * m + s


def clock(sec: float) -> str:
    """Seconds since midnight -> hh.mm.ss (the lists' own format)."""
    sec = int(round(sec)) % 86400
    return f"{sec // 3600:02d}.{sec % 3600 // 60:02d}.{sec % 60:02d}"


def _parse_time(value: str, where: str, choose: int | None = None) -> int:
    """The single clock time in `value`, in seconds since midnight.

    Separators may be '.' or ':' and mixed (16:13.23). Text in parentheses and the
    word 'oppure' are allowed around the times. Anything else that contains a digit
    raises, so a damaged time ("1 6.49.25") cannot be half-read. If the value holds
    more than one time, `choose` (from DECISIONS) must say which.
    """
    tokens = [m.group(0) for m in _TIME.finditer(value)]
    rest = _TIME.sub("", value)
    rest = re.sub(r"\([^)]*\)|\bop+ure\b|[;,]", "", rest, flags=re.I).strip()
    if not tokens or re.search(r"\d", rest):
        raise ValueError(f"{where}: cannot read a time from {value!r}")
    if len(tokens) > 1 and choose is None:
        raise ValueError(f"{where}: {len(tokens)} times in {value!r} and no decision "
                         f"says which to use")
    return _seconds(tokens[choose or 0])


@dataclass
class ListedSeizure:
    """One seizure as listed, times in seconds since midnight (wall clock)."""
    subject: str
    number: int                     # the list's own seizure number ("Seizure n 3")
    file: str
    reg_start: int
    reg_end: int
    start: int
    end: int
    inherited: bool = False         # registration times taken from an earlier block
    notes: list[str] = field(default_factory=list)


def _decide(subject: str, number: int, what: str, decisions: dict) -> int | None:
    """Index of the time to use in a multi-time value, per DECISIONS."""
    if (subject, number, what) == ("PN10", 3, "start"):
        return {"clinical": 0, "electric": 1}[decisions["PN10_seizure3_onset"]]
    if (subject, number, what) == ("PN10", 2, "end"):
        return {"first": 0, "second": 1}[decisions["PN10_seizure2_end"]]
    return None


def check_decisions(decisions: dict) -> None:
    for k, opts in _DECISION_OPTIONS.items():
        if decisions.get(k) not in opts:
            raise ValueError(f"DECISIONS[{k!r}] = {decisions.get(k)!r}, expected one "
                             f"of {opts}")


def parse_seizure_list(subject: str, root: Path = SIENA_ROOT,
                       decisions: dict | None = None) -> list[ListedSeizure]:
    """All seizures of Seizures-list-<subject>.txt, with CORRECTIONS applied.

    Parsing is sequential, as for CHB-MIT: "File name" and "Registration ... time"
    lines set the current recording context, "Seizure n K" opens a seizure, and its
    start and end lines fill it. This handles both layouts in the data: the usual one
    (context after "Seizure n K") and PN01's (context before "Seizure n 1", shared by
    both seizures).
    """
    decisions = DECISIONS if decisions is None else decisions
    check_decisions(decisions)
    path = root / subject / f"Seizures-list-{subject}.txt"
    text = path.read_text(encoding="latin-1")
    for old, new in CORRECTIONS["text"].get(subject, []):
        if old not in text:
            raise ValueError(f"{path.name}: correction {old!r} -> {new!r} no longer "
                             f"applies; the list has changed")
        text = text.replace(old, new)

    out: list[ListedSeizure] = []
    ctx = {"file": None, "reg_start": None, "reg_end": None}
    reg_seen = False                # registration lines seen since the last seizure
    cur: dict | None = None

    def close() -> None:
        nonlocal cur, reg_seen
        if cur is None:
            return
        n = cur["number"]
        where = f"{subject} seizure {n}"
        if cur.get("start") is None or cur.get("end") is None:
            raise ValueError(f"{where}: start or end time missing")
        if ctx["file"] is None or ctx["reg_start"] is None or ctx["reg_end"] is None:
            raise ValueError(f"{where}: no file name or registration times")
        inherited = not reg_seen
        if inherited and (subject, n) not in CORRECTIONS["inherited_registration"]:
            raise ValueError(f"{where}: no registration times of its own; add it to "
                             f"CORRECTIONS['inherited_registration'] if intended")
        out.append(ListedSeizure(subject, n, ctx["file"], ctx["reg_start"],
                                 ctx["reg_end"], cur["start"], cur["end"], inherited))
        cur, reg_seen = None, False

    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"Seizure n\s*(\d+)", line, re.I)
        if m:
            close()
            cur = {"number": int(m.group(1)), "start": None, "end": None}
            continue
        m = re.match(r"File name:\s*(\S+)", line, re.I)
        if m:
            name = CORRECTIONS["file_names"].get(m.group(1), m.group(1))
            if name != ctx["file"]:
                ctx = {"file": name, "reg_start": None, "reg_end": None}
            continue
        m = re.match(r"Registration (start|end) time:\s*(.*)", line, re.I)
        if m:
            ctx["reg_" + m.group(1).lower()] = _parse_time(
                m.group(2), f"{subject} {ctx['file']} registration {m.group(1)}")
            reg_seen = True
            continue
        m = re.match(r"(?:Seizure )?(start|end) time:\s*(.*)", line, re.I)
        if m:
            what = m.group(1).lower()
            if cur is None:
                raise ValueError(f"{subject}: seizure {what} time outside a seizure "
                                 f"block: {line!r}")
            cur[what] = _parse_time(m.group(2), f"{subject} seizure {cur['number']} "
                                    f"{what}", _decide(subject, cur["number"], what,
                                                       decisions))
    close()
    if not out:
        raise ValueError(f"{path.name}: no seizures parsed")

    for z in out:
        if (z.subject, z.number) == ("PN00", 3) and z.end == _seconds("19.29.29"):
            z.notes.append("listed end 19.29.29 is after the recording end 18.57.13")
            if decisions["PN00_seizure3"] == "typo":
                z.end = _seconds("18.29.29")
                z.notes.append("read as 18.29.29 (DECISIONS: typo)")
            else:
                z.notes.append("excluded with its windows (DECISIONS: exclude)")
    return out


def excluded(z: ListedSeizure, decisions: dict | None = None) -> bool:
    """True for a seizure that DECISIONS removes from the corpus."""
    decisions = DECISIONS if decisions is None else decisions
    return (z.subject, z.number) == ("PN00", 3) and decisions["PN00_seizure3"] == "exclude"


# --- EDF header details CHB-MIT does not need --------------------------------------------

def recording_start(path: Path) -> int:
    """Start time of the recording from the EDF header (bytes 176-183, hh.mm.ss), in
    seconds since midnight.

    This, not the list's "Registration start time", is the reference for seizure
    times: the EDF start time belongs to the first sample by definition, and the
    header audit (`python -m src.siena_download --audit`) shows that the two agree for
    39 of 41 recordings and disagree for PN14-3 (list three hours early, a typo: the
    header start plus the duration gives the listed end exactly) and PN05-3 (10 s,
    unresolved; see CORRECTIONS).
    """
    with open(path, "rb") as fh:
        raw = fh.read(256)
    return _seconds(raw[176:184].decode("ascii").strip())


def physical_dimensions(path: Path) -> list[str]:
    """Physical dimension (unit) of every signal, from the EDF header."""
    h = read_edf_header(path)
    ns = len(h.labels)
    with open(path, "rb") as fh:
        fh.seek(256 + ns * (16 + 80))
        buf = fh.read(8 * ns).decode("ascii", "replace")
    return [buf[i * 8:(i + 1) * 8].strip() for i in range(ns)]


def electrode_index(labels: list[str], where: str = "") -> dict[str, int]:
    """Position of each of the 19 needed electrodes among the EDF labels; raises if any
    is missing or appears twice."""
    norm = [electrode_label(l) for l in labels]
    out = {}
    for e in ELECTRODES:
        hits = [i for i, l in enumerate(norm) if l == e]
        if len(hits) != 1:
            raise ValueError(f"{where}: electrode {e} found {len(hits)} times "
                             f"among the EDF labels")
        out[e] = hits[0]
    return out


def seizures_in_record(path: Path, listed: list[ListedSeizure],
                       decisions: dict | None = None
                       ) -> tuple[list[Seizure], list[tuple[float, float]], list[str]]:
    """Seizure intervals of one recording in seconds from its first sample.

    Returns (seizures, excluded intervals, seizure ids). Wall-clock times are converted
    relative to the EDF header start time, modulo 24 hours, which handles recordings
    that run past midnight (PN01, PN03, PN06-2, PN07, PN14-3, PN16-2). Each interval is
    checked to lie inside the recording.
    """
    h = read_edf_header(path)
    t_start = recording_start(path)
    zs, excl, ids = [], [], []
    k = 0
    for z in listed:
        if z.file != path.name:
            continue
        s = (z.start - t_start) % 86400
        e = (z.end - t_start) % 86400
        if excluded(z, decisions):
            excl.append((float(s), float(h.duration)))
            continue
        if not (0 <= s < e <= h.duration):
            raise ValueError(f"{path.name}: seizure {z.number} at {s}-{e} s does not "
                             f"lie inside the recording (0-{h.duration} s)")
        k += 1
        zs.append(Seizure(path.name, float(s), float(e)))
        ids.append(f"{path.name}#{k}")
    return zs, excl, ids


# --- Signal --------------------------------------------------------------------------------

def read_bipolar(path: Path, signal_fn=None) -> np.ndarray:
    """A recording as the 18 TARGET_CHANNELS bipolar signals at 256 Hz, in microvolts.

    Shape (18, samples). Only the 19 needed electrodes are scaled to float, which keeps
    the longest recording (PN14-3, 11.7 hours, 49 signals, 2.1 GB) within a few GB of
    memory. Downsampling 512 -> 256 Hz uses resample_poly (polyphase, with its own
    anti-aliasing filter) on the continuous recording, before any windowing, so no
    window edge sees a filter transient.

    signal_fn(x, fs, key), if given, is applied to the continuous 256 Hz bipolar signal,
    as in CHB-MIT (src/chbmit.py: extract_windows).
    """
    h = read_edf_header(path)
    if len(set(h.samples_per_record)) != 1:
        raise ValueError(f"{path.name}: channels have different sampling rates")
    if abs(h.fs[0] - FS_SIENA) > 1e-6:
        raise ValueError(f"{path.name}: sampling rate {h.fs[0]}, expected {FS_SIENA}")
    idx = electrode_index(h.labels, path.name)
    dims = physical_dimensions(path)
    bad = {e: dims[i] for e, i in idx.items() if dims[i].lower() not in ("uv", "µv")}
    if bad:
        raise ValueError(f"{path.name}: electrodes not in microvolts: {bad}")

    ns, spr = len(h.labels), h.samples_per_record[0]
    raw = np.memmap(path, dtype="<i2", mode="r", offset=h.header_bytes,
                    shape=(h.n_records, ns, spr))
    order = [idx[e] for e in ELECTRODES]
    sig = np.empty((len(order), h.n_records * spr), np.float32)
    for j, i in enumerate(order):
        d = raw[:, i, :].reshape(-1).astype(np.float32)
        gain = (h.phys_max[i] - h.phys_min[i]) / ((h.dig_max[i] - h.dig_min[i]) or 1.0)
        sig[j] = (d - h.dig_min[i]) * gain + h.phys_min[i]
    del raw

    pos = {e: j for j, e in enumerate(ELECTRODES)}
    x = np.stack([sig[pos[a]] - sig[pos[b]] for a, b in bipolar_pairs()])
    del sig
    x = resample_poly(x, 1, int(FS_SIENA // FS_CHB), axis=-1).astype(np.float32)
    if signal_fn is not None:
        x = signal_fn(x, FS_CHB, path.name)
    return x


def record_path(subject: str, name: str, root: Path = SIENA_ROOT) -> Path:
    return root / subject / name


def subject_records(subject: str, root: Path = SIENA_ROOT) -> list[str]:
    """Recording file names of a subject, from RECORDS (the authoritative list)."""
    recs = (root / "RECORDS").read_text().split()
    return [r.split("/")[1] for r in recs if r.startswith(subject + "/")]
