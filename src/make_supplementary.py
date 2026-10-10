"""Supplementary tables from stored results: paper/supplementary.md.

Every number comes from a file in results_v2/ (or, for the P4 table, from the corpus,
which it then stores in results_v2/qc/ so later builds need no raw data):

  S1  artefact removal (P6a/b/c): signal retention and compute per recording
  S2  GEDAI preset choice, by signal preservation only
  S3  P4 technical artefact rejection per person, ictal and non-ictal separately

Usage:  python -m src.make_supplementary
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import PROJECT_ROOT, RESULTS_ROOT

QC = RESULTS_ROOT / "qc"
OUT = PROJECT_ROOT / "paper" / "supplementary.md"


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("" if isinstance(v, float) and not np.isfinite(v) else str(v))
                 for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_s1() -> str:
    d = pd.read_csv(QC / "p6_recording_logs.csv")
    names = {"infomax": "P6a Extended Infomax", "gedai-conservative": "P6b GEDAI, conservative (used)",
             "amica": "P6c AMICA", "gedai": "GEDAI, default (not used)"}
    rows = []
    for m in ("infomax", "gedai-conservative", "amica", "gedai"):
        g = d[d.method == m]
        v = g.power_kept
        rows.append({"method": names[m], "recordings": len(g),
                     "power kept, median": v.median(),
                     "IQR": f"{v.quantile(.25):.3f}-{v.quantile(.75):.3f}",
                     "min": v.min(), "share < 0.5": (v < 0.5).mean(),
                     "components removed, mean": g.n_removed.mean() if g.n_removed.notna().any()
                     else np.nan,
                     "seconds per recording, median": f"{g.seconds.median():.0f}",
                     "CPU hours, total": f"{g.seconds.sum() / 3600:.1f}"})
    return md(pd.DataFrame(rows))


def table_s2() -> str:
    t = pd.read_csv(QC / "gedai_settings" / "summary.csv")
    t = t.rename(columns={"setting": "preset", "unit": "unit", "share_below_0.5": "share < 0.5",
                          "share_below_0.1": "share < 0.1"})
    t["preset"] = t.preset.map({"auto": "default (auto)",
                                "auto-": "conservative (auto-), used"}).fillna(t.preset)
    return md(t[["preset", "unit", "n", "p5", "p25", "median", "p75", "share < 0.5", "share < 0.1"]])


def table_s3() -> str:
    f = QC / "p4_rejection_per_person.csv"
    if not f.exists():
        from .chbmit_corpus import build_corpus
        from .preprocess import artifact_mask
        d = build_corpus(verbose=False)
        rej = artifact_mask(d["X"])
        g = pd.DataFrame({"person": d["group"], "ictal": d["y"] == 1, "rejected": rej})
        t = g.groupby("person")[["ictal", "rejected"]].apply(lambda x: pd.Series({
            "windows": len(x), "ictal": int(x.ictal.sum()),
            "ictal rejected": int((x.ictal & x.rejected).sum()),
            "non-ictal": int((~x.ictal).sum()),
            "non-ictal rejected": int((~x.ictal & x.rejected).sum())}))
        t = t.reset_index()
        t.to_csv(f, index=False)
    t = pd.read_csv(f)
    tot = t.drop(columns="person").sum()
    t.loc[len(t)] = ["all", *tot.values]
    t["ictal rejected %"] = 100 * t["ictal rejected"] / t.ictal
    t["non-ictal rejected %"] = 100 * t["non-ictal rejected"] / t["non-ictal"]
    t[["windows", "ictal", "ictal rejected", "non-ictal", "non-ictal rejected"]] = \
        t[["windows", "ictal", "ictal rejected", "non-ictal", "non-ictal rejected"]].astype(int)
    return md(t, "{:.1f}")


# Siena metadata decisions (docs/SIENA.md, src/siena.py: DECISIONS and CORRECTIONS). The
# window counts were computed with siena_corpus.build_corpus under each alternative
# (10 October 2026); every ictal window is kept in the corpus, so the ictal counts compare
# directly.
SIENA_DECISIONS = [
    ("PN00", "PN00-3, seizure 3", "listed end 19.29.29 lies 1916 s after the recording ends (18.57.33)",
     "end read as 18.29.29 (a 60 s seizure)", "verified correction",
     "signal: the right frontotemporal 3-20 Hz ictal rhythm (FP2-F8, F8-T8) falls below a quarter of "
     "its peak 14 s after 18.29.29; the patient's other four seizures end 0 to 13 s after their "
     "listed ends by the same rule",
     "offset from an automatic rule on one channel pair",
     "alternative (exclude the seizure and the rest of the recording): 30 instead of 37 ictal "
     "windows for PN00"),
    ("PN05", "PN05-3, seizure 3", "registration start: list 06.01.23, EDF header 06.01.13",
     "EDF header start time, as for every recording", "decision rule (not a verified correction)",
     "none possible: the header defines the first sample and 39 of 41 recordings agree with their "
     "list; the list's value fits the 20 s end offset of 18 other recordings, the header's does not",
     "10 s in the seizure's position",
     "4 ictal windows either way, shifted by one: 2 window labels differ"),
    ("PN10", "PN10-3, seizure 3", "two onsets: 15.43.53 (clinical), 15.43.59 (electrical)",
     "electrical onset (advisor)", "decision",
     "CHB-MIT annotates electrographic onsets; the electrical onset keeps the two corpora consistent",
     "none (both listed)",
     "clinical onset instead: 48 instead of 47 ictal windows for PN10 (1 label)"),
    ("PN10", "PN10-4.5.6, seizure 6", "only a clinical onset (15.18.26)",
     "used as listed, flagged in the corpus's seizure notes", "decision",
     "-", "electrical onset unknown; in seizure 3 it followed the clinical one by 6 s",
     "at most about one window, by analogy with seizure 3"),
    ("PN10", "PN10-2, seizure 2", "end \"11.41.04 opure 11.40.43\" (Italian \"or\")",
     "first value, 11.41.04", "decision rule", "-", "21 s in the seizure's end",
     "second value instead: 45 instead of 47 ictal windows for PN10"),
    ("PN14", "PN14-3", "registration start: list 16.17.45, EDF header 19.17.45",
     "EDF header start time", "verified correction",
     "header start + duration (41,995 s) gives the listed end time 06.57.40 exactly; the list's "
     "start would make the recording three hours longer than the file",
     "none", "seizure at 1 h 52 min into the recording, not 4 h 52 min"),
]


def table_s4() -> str:
    t = pd.DataFrame(SIENA_DECISIONS, columns=[
        "patient", "recording / seizure", "problem in the metadata", "used", "type",
        "how it was checked", "remaining uncertainty", "effect on the windows"])
    return md(t)


def main() -> None:
    text = f"""# Supplementary Material

*Generated by `python -m src.make_supplementary` from `results_v2/qc/`; do not edit by hand.*

## Table S1. Artefact removal in P6: signal retention and compute

Per recording of CHB-MIT with selected windows (670), on the 0.5 to 40 Hz signal. Power
kept = signal power after removal / before. Extended Infomax and AMICA share 16
components, fitting on every 8th sample and the same automatic ocular component rule;
GEDAI selects what to remove by its own criterion. Compute on a laptop CPU (Intel Core
Ultra 7 155U), six recordings in parallel.

{table_s1()}

## Table S2. GEDAI preset, chosen on signal preservation only

Six CHB-MIT subjects (chb01, chb03, chb05, chb08, chb12, chb15), the same windows under
both presets. Power kept per recording and per window (relative to the same 0.5 to 40 Hz
signal); no classification result was used for the choice (`src/gedai_qc.py`).

{table_s2()}

## Table S3. P4: technical artefact rejection per person

A window is rejected if a channel is flat (standard deviation below 1 uV) or holds one
constant value for at least 0.5 s, on the unfiltered signal. Rejected windows are removed
from training and validation only; test windows are unchanged. Rejection rates are given
for ictal and non-ictal windows separately. chb01 includes chb21, the same person.

{table_s3()}

## Table S4. Siena: metadata problems and the decisions taken

Every departure from the published seizure lists, with the decision used in the main
analysis, its kind (a correction verified against the recording, or a decision rule
applied without such a check), how it was checked, and how much the alternative would
change the windows. Other, purely textual fixes (a space inside a time in PN10's list,
misspelt file names, PN12's missing registration times taken from the same file's
first block, recordings past midnight read modulo 24 h) are listed in `docs/SIENA.md`,
Section 4. Recording time is taken from the EDF headers, 141.0 h; the dataset's
description gives 128.4 h, a difference explained for PN14 (the start time above) but
not for PN03, PN10 and PN12, whose files contain no padding (`docs/SIENA.md`, Section 8).

{table_s4()}
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
