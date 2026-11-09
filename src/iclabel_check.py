"""How far does P6's ocular component rule overlap with ICLabel? One Siena patient.

P6a and P6c remove the components that MNE's find_bads_eog (|z| > 3) links to the four
frontal bipolar channels (FP1-F7, FP1-F3, FP2-F4, FP2-F8), since neither corpus has an EOG
channel (src/preprocess.py). This script checks that rule against ICLabel (Pion-Tonachini
et al. 2019, mne-icalabel), which classifies each component from its topography,
spectrum and autocorrelation into brain, muscle, eye, heart, line noise, channel noise
or other.

ICLabel needs electrode positions, a common average reference, a 1 to 100 Hz band-pass
and an extended Infomax decomposition (mne-icalabel documentation). The P6 decompositions
themselves cannot be labelled: they are fitted on the 18 bipolar channels at 0.5 to 40
Hz. CHB-MIT was recorded in the bipolar montage: the 18 channels used in the paper form
three unconnected electrode groups (left chains, right chains, midline). Recordings that
also carry FT9-FT10 link the left and right groups, so an exact common average over the
18 lateral electrodes exists there (without FZ, CZ, PZ); that is not done here. Siena was recorded
referentially, so here each recording is read with all its EEG electrodes and prepared
as ICLabel requires:

    EEG electrodes only (29 in PN00), standard 10-05 positions
    notch at 50 and 100 Hz, band-pass 1 to 100 Hz, common average reference
    extended Infomax, rank - 1 components (the average reference removes one),
    fitted on every ICA_FIT_DECIM-th sample, seeded from the file name as in P6

Both labellings are then applied to the SAME decomposition:

    P6 rule   find_bads_eog with the four frontal bipolar derivations as EOG proxies,
              formed as in the paper (raw referential signal, then 0.5 to 40 Hz), and
              the same threshold (|z| > EOG_Z)
    ICLabel   the most probable of the seven classes, with all seven probabilities

For every component the script also stores its explained variance and its power during
seizures relative to the rest of the recording, broadband and in 30 to 100 Hz, because
P6 deliberately keeps muscle components (the gamma band carries the strongest ictal
signal): if ICLabel calls ictal components "muscle", a removal rule based on ICLabel's
muscle class would delete seizure activity.

This is a separate decomposition, not the P6 one: the result says how far the P6 rule
overlaps with ICLabel on an ICLabel-conformant ICA, not that the P6 components were
validated. With one patient it is descriptive.

The patient is chosen from subject_info.csv alone, before any result: the patient with
the most seizures per recording hour among those with the full 29 electrodes
(pick_subject; PN00), so that the short run covers as many seizures as possible.
--lateralization L picks the same way among left-sided patients.

Usage:
    .venv-ica/Scripts/python -m src.iclabel_check                  # pick_subject(), all recordings
    .venv-ica/Scripts/python -m src.iclabel_check --lateralization L
    .venv-ica/Scripts/python -m src.iclabel_check --subject PN05 --records PN05-2.edf
Output: results_v2/iclabel/<subject>/components.csv, summary.md, meta.json,
        topomaps_<record>.png
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .config import PROJECT_ROOT
from .preprocess import EOG_PROXY, EOG_Z, ICA_FIT_DECIM, ICA_MAX_ITER, bandpass
from .siena import (OLD_NAMES, SIENA_ROOT, parse_seizure_list, record_path,
                    seizures_in_record, subject_records)

OUT_ROOT = PROJECT_ROOT / "results_v2" / "iclabel"
BAND = (1.0, 100.0)               # ICLabel's training preprocessing
P6_BAND = (0.5, 40.0)             # band of the bipolar signal the P6 rule was applied to
NOTCH = (50.0, 100.0)             # Italian mains and its harmonic (inside the band)
GAMMA = (30.0, 100.0)
CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise",
           "channel noise", "other")
NEW_NAMES = {old: new for new, old in OLD_NAMES.items()}     # T3 -> T7, ...
SELECTION = ("the patient with the most seizures per recording hour among those with "
             "the full 29 electrodes (subject_info.csv), so that a short run covers as "
             "many seizures as possible.")


def pick_subject(lateralization: str | None = None) -> str:
    """Most seizures per recording hour among the patients with 29 EEG electrodes, from
    subject_info.csv only (no result is looked at)."""
    info = pd.read_csv(SIENA_ROOT / "subject_info.csv", skipinitialspace=True)
    info = info[info["eeg_channel"] == 29]
    if lateralization:
        info = info[info["lateralization"] == lateralization]
    rate = info["number_seizures"] / info["rec_time_minutes"]
    return str(info.loc[rate.idxmax(), "patient_id"])


def read_referential(path):
    """All EEG electrodes of a Siena recording as an MNE Raw with 10-05 positions."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    eeg = [c for c in raw.ch_names if c.upper().startswith("EEG ")]
    raw.pick(eeg).load_data()
    montage = mne.channels.make_standard_montage("standard_1005")
    by_upper = {c.upper(): c for c in montage.ch_names}
    rename = {}
    for c in eeg:
        name = c[4:].strip()
        name = NEW_NAMES.get(name.upper(), name)
        if name.upper() not in by_upper:
            raise ValueError(f"{path.name}: electrode {c} has no 10-05 position")
        rename[c] = by_upper[name.upper()]
    raw.rename_channels(rename)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    raw.set_montage(montage)
    return raw


def prepare(raw):
    """The four frontal bipolar derivations as EOG channels for the P6 rule, formed as
    in the paper: from the raw referential signal, then the P6 band-pass (0.5 to 40 Hz,
    src/preprocess.py: bandpass). Then ICLabel's preprocessing of the EEG channels."""
    import mne
    fs = raw.info["sfreq"]
    x = {c.upper(): raw.get_data(picks=[c])[0] for c in raw.ch_names}
    names, rows = [], []
    for ch in EOG_PROXY:
        a, b = (NEW_NAMES.get(e, e) for e in ch.split("-"))
        names.append(ch)
        rows.append(x[a.upper()] - x[b.upper()])
    rows = bandpass(np.array(rows) * 1e6, fs, *P6_BAND).astype(np.float64) * 1e-6
    raw.notch_filter(NOTCH, verbose=False)
    raw.filter(*BAND, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    info = mne.create_info(names, fs, ch_types="eog", verbose=False)
    eog = mne.io.RawArray(rows, info, verbose=False)
    raw.add_channels([eog], force_update_info=True)
    return raw


def ictal_mask(path, subject, n, fs):
    """True on seizure samples; excluded intervals (src/siena.py: DECISIONS) are
    returned separately so they count as neither seizure nor background."""
    zs, excl, _ = seizures_in_record(path, parse_seizure_list(subject))
    ictal = np.zeros(n, bool)
    skip = np.zeros(n, bool)
    for z in zs:
        ictal[int(z.start_s * fs):int(z.end_s * fs)] = True
    for s, e in excl:
        skip[int(s * fs):int(e * fs)] = True
    return ictal, skip, len(zs)


def check_record(subject, name, out):
    import mne
    from mne_icalabel.iclabel import iclabel_label_components
    mne.set_log_level("ERROR")
    t0 = time.time()
    path = record_path(subject, name)
    raw = prepare(read_referential(path))
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    ica = mne.preprocessing.ICA(n_components=n_eeg - 1, method="infomax",
                                fit_params=dict(extended=True),
                                max_iter=ICA_MAX_ITER["infomax"],
                                random_state=zlib.crc32(name.encode()) % (2 ** 32))
    ica.fit(raw, picks="eeg", decim=ICA_FIT_DECIM)

    p6: dict[int, float] = {}
    for ch in EOG_PROXY:
        idx, scores = ica.find_bads_eog(raw, ch_name=ch, threshold=EOG_Z)
        for i in idx:
            p6[int(i)] = max(p6.get(int(i), 0.0), abs(float(scores[i])))

    eeg = raw.copy().pick("eeg")
    proba = iclabel_label_components(eeg, ica, inplace=False)

    fs = raw.info["sfreq"]
    src = ica.get_sources(eeg).get_data()
    ictal, skip, n_seizures = ictal_mask(path, subject, src.shape[1], fs)
    bg = ~ictal & ~skip
    sos = butter(4, GAMMA, btype="bandpass", fs=fs, output="sos")
    gamma = sosfiltfilt(sos, src, axis=-1)

    rows = []
    for i in range(ica.n_components_):
        ev = ica.get_explained_variance_ratio(eeg, components=[i], ch_type="eeg")["eeg"]
        row = {"record": name, "component": i,
               "iclabel": CLASSES[int(np.argmax(proba[i]))],
               "iclabel_p": float(proba[i].max()),
               "p6_rule": i in p6, "p6_max_abs_z": p6.get(i, np.nan),
               "explained_variance": float(ev)}
        row |= {f"p_{c.split()[0]}": float(p) for c, p in zip(CLASSES, proba[i])}
        if ictal.any():
            row["ictal_power_ratio"] = float((src[i, ictal] ** 2).mean()
                                             / (src[i, bg] ** 2).mean())
            row["ictal_gamma_ratio"] = float((gamma[i, ictal] ** 2).mean()
                                             / (gamma[i, bg] ** 2).mean())
        rows.append(row)
    plot_topomaps(ica, eeg, rows, out / f"topomaps_{name.removesuffix('.edf')}.png")
    print(f"{name}: {ica.n_components_} components, {n_seizures} seizures, "
          f"P6 rule {sorted(p6)}, ICLabel eye "
          f"{[r['component'] for r in rows if r['iclabel'] == 'eye blink']}, "
          f"{time.time() - t0:.0f} s", flush=True)
    return rows


def plot_topomaps(ica, eeg, rows, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    maps = ica.get_components()
    n = maps.shape[1]
    cols = 7
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(cols * 1.9, -(-n // cols) * 2.2))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, ax in zip(range(n), axes.ravel()):
        mne.viz.plot_topomap(maps[:, i], eeg.info, axes=ax, show=False, contours=0)
        r = rows[i]
        mark = " [P6]" if r["p6_rule"] else ""
        ax.set_title(f"IC{i}{mark}\n{r['iclabel'].split()[0]} {r['iclabel_p']:.2f}",
                     fontsize=8, color="crimson" if r["p6_rule"] else "black")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def summarise(df, subject, selection):
    eye = df["iclabel"] == "eye blink"
    p6 = df["p6_rule"]
    lines = [f"# Overlap of the P6 ocular rule with ICLabel: Siena {subject}", "",
             "Generated by `python -m src.iclabel_check`; do not edit by hand.", "",
             f"{df['record'].nunique()} recordings, {len(df)} components of an "
             f"ICLabel-conformant ICA (extended Infomax, 1 to 100 Hz, common average "
             f"reference). This is a separate decomposition from the P6 one, so the "
             f"table shows how far the P6 rule overlaps with ICLabel here, not a "
             f"validation of the P6 components. One patient: descriptive only.", "",
             f"Patient choice: {selection}", "",
             "## Overlap on the same decomposition", "",
             "| | ICLabel eye | ICLabel other class | total |", "|---|---|---|---|",
             f"| P6 rule removes | {(p6 & eye).sum()} | {(p6 & ~eye).sum()} | {p6.sum()} |",
             f"| P6 rule keeps | {(~p6 & eye).sum()} | {(~p6 & ~eye).sum()} | {(~p6).sum()} |",
             "",
             "ICLabel class of the components the P6 rule removes:", "",
             "| record | component | ICLabel | p | p(eye) | explained variance |",
             "|---|---|---|---|---|---|"]
    for _, r in df[p6].iterrows():
        lines.append(f"| {r['record']} | {r['component']} | {r['iclabel']} | "
                     f"{r['iclabel_p']:.2f} | {r['p_eye']:.2f} | {r['explained_variance']:.3f} |")
    lines += ["", "## ICLabel classes", "",
              "| class | components | median ictal gamma ratio |", "|---|---|---|"]
    for c in CLASSES:
        sub = df[df["iclabel"] == c]
        if len(sub):
            g = sub["ictal_gamma_ratio"].median() if "ictal_gamma_ratio" in sub else np.nan
            lines.append(f"| {c} | {len(sub)} | {g:.2f} |")
    if "ictal_gamma_ratio" in df:
        top = df.sort_values("ictal_gamma_ratio", ascending=False).head(10)
        lines += ["", "## Components with the largest ictal increase in 30 to 100 Hz", "",
                  "Ratio of mean power during seizures to the rest of the recording.", "",
                  "| record | component | ICLabel | p | gamma ratio | broadband ratio | P6 rule |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(f"| {r['record']} | {r['component']} | {r['iclabel']} | "
                         f"{r['iclabel_p']:.2f} | {r['ictal_gamma_ratio']:.2f} | "
                         f"{r['ictal_power_ratio']:.2f} | {'removes' if r['p6_rule'] else 'keeps'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", help="default: pick_subject()")
    ap.add_argument("--lateralization", choices=["L", "R"],
                    help="pick_subject() among patients with this seizure side")
    ap.add_argument("--records", nargs="*", help="default: all recordings of the subject")
    ap.add_argument("--selection", help="why --subject was chosen (written to summary.md)")
    args = ap.parse_args()
    import mne
    import mne_icalabel
    subject = args.subject or pick_subject(args.lateralization)
    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    records = args.records or subject_records(subject)
    rows = []
    for name in records:
        part = out / f"components_{name.removesuffix('.edf')}.csv"   # checkpoint: reruns resume
        if part.exists():
            rows += pd.read_csv(part).to_dict("records")
            print(f"{name}: already done, reusing {part.name}", flush=True)
            continue
        recs = check_record(subject, name, out)
        pd.DataFrame(recs).to_csv(part, index=False)
        rows += recs
    df = pd.DataFrame(rows)
    df.to_csv(out / "components.csv", index=False)
    if args.subject:
        selection = args.selection or "given with --subject"
    else:
        side = {"L": "left", "R": "right"}.get(args.lateralization)
        selection = (f"among the {side}-sided patients, " if side else "") + SELECTION
        selection = selection[0].upper() + selection[1:] + " Chosen before any result."
    (out / "summary.md").write_text(summarise(df, subject, selection), encoding="utf-8")
    meta = {"subject": subject, "selection": selection,
            "lateralization": args.lateralization,
            "records": records, "band": BAND, "notch": NOTCH, "eog_band": P6_BAND,
            "reference": "average", "ica": "extended infomax",
            "fit_decim": ICA_FIT_DECIM, "max_iter": ICA_MAX_ITER["infomax"],
            "eog_proxy": list(EOG_PROXY), "eog_z": EOG_Z, "gamma": GAMMA,
            "mne": mne.__version__, "mne_icalabel": mne_icalabel.__version__,
            "python": platform.python_version(), "cpu": platform.processor()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
