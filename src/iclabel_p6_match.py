"""Did P6 itself remove what the ICLabel check found? Matching the two decompositions.

src/iclabel_check.py applies the P6 ocular rule to a separate, ICLabel-conformant ICA
(29 referential electrodes, 1 to 100 Hz, average reference). A component the rule flags
there is not necessarily removed by P6, which decomposes the 18 bipolar channels at
0.5 to 40 Hz. This script asks, for every ICLabel component of a recording: how much of
its signal does the actual P6 cleaning remove?

    1. P6 is recomputed exactly as the cache was built (src/siena.py: read_bipolar, then
       src/preprocess.py: bandpass and the Infomax branch of ica_clean, seeded from the
       file name). The removed components and the retained power are compared with the
       log written when the cache was built (data/ica_logs/bp0.5-40_infomax), because
       Infomax can differ between execution contexts (docs/ICA_REPRO.md). Only
       recordings that reproduce the log are interpreted.
    2. The ICLabel ICA is recomputed (src/iclabel_check.py) and its classes are compared
       with the stored components_<record>.csv.
    3. Each ICLabel component's own contribution to the scalp signal is formed (the
       recording reconstructed from that component alone), converted to the 18 bipolar
       channels, resampled to 256 Hz and band-passed 0.5 to 40 Hz like the P6 input.
       P6's removal is linear, so the share of that contribution's power that P6
       removes is well defined: removed_share in [0, 1].

An eye component that P6 removes should show a high removed_share; a brain component
P6 leaves alone a low one. The ICA must run with nothing else heavy on the machine, on
the machine that built the P6 caches (docs/ICA_REPRO.md).

Usage:
    .venv-ica/Scripts/python -W ignore -m src.iclabel_p6_match --subject PN00 --records PN00-2.edf PN00-5.edf
Output: results_v2/iclabel/<subject>/p6_match_<record>.csv, p6_match.md
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import FS_CHB, TARGET_CHANNELS
from .iclabel_check import CLASSES, NEW_NAMES, OUT_ROOT, prepare, read_referential
from .preprocess import (EOG_PROXY, EOG_Z, ICA_FIT_DECIM, ICA_LOG_ROOT, ICA_MAX_ITER,
                         ICA_N_COMPONENTS, _mne_raw, bandpass)
from .siena import read_bipolar, record_path

P6_BAND = (0.5, 40.0)
P6_LOG = ICA_LOG_ROOT / "bp0.5-40_infomax"


def p6_ica(path):
    """The P6a decomposition of one recording, as ica_clean builds it, without writing
    a log. Returns the fitted ICA, the excluded components and the retained power."""
    import mne
    mne.set_log_level("ERROR")
    x = bandpass(read_bipolar(path), FS_CHB, *P6_BAND)
    raw = _mne_raw(x, FS_CHB)
    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method="infomax",
                                fit_params=dict(extended=True),
                                max_iter=ICA_MAX_ITER["infomax"],
                                random_state=zlib.crc32(path.name.encode()) % (2 ** 32))
    ica.fit(raw, decim=ICA_FIT_DECIM)
    bad: set[int] = set()
    for ch in EOG_PROXY:
        idx, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=EOG_Z)
        bad |= {int(i) for i in idx}
    ica.exclude = sorted(bad)
    out = ica.apply(raw.copy()).get_data()
    xd = x.astype(np.float64) * 1e-6
    power_kept = float((out ** 2).sum() / (xd ** 2).sum())
    return ica, ica.exclude, power_kept


def p6_removed(ica, y):
    """The part of a bipolar signal y (18, T), volts, that P6's removal takes out.
    ica.apply is affine (it subtracts and restores the PCA mean), so the constant part
    is removed by subtracting the response to a zero signal."""
    import mne
    info = mne.create_info(list(TARGET_CHANNELS), FS_CHB, ch_types="eeg", verbose=False)

    def removed(z):
        r = mne.io.RawArray(z.copy(), info, verbose=False)    # RawArray does not copy, and
        return z - ica.apply(r).get_data()                    # apply works in place
    return removed(y) - removed(np.zeros((y.shape[0], 1)))


def to_bipolar(ref, names):
    """Referential electrodes (n, T) -> the 18 TARGET_CHANNELS bipolar signals."""
    pos = {n.upper(): i for i, n in enumerate(names)}
    rows = []
    for ch in TARGET_CHANNELS:
        a, b = (NEW_NAMES.get(e.upper(), e).upper() for e in ch.split("-"))
        rows.append(ref[pos[a]] - ref[pos[b]])
    return np.array(rows)


def match_record(subject, name, out):
    import mne
    from mne_icalabel.iclabel import iclabel_label_components
    path = record_path(subject, name)

    stem = name.removesuffix(".edf")
    keep = out / "ica"                       # fitted decompositions, so reruns skip the fits
    keep.mkdir(exist_ok=True)
    p6_file, p6_meta = keep / f"p6a_{stem}-ica.fif", keep / f"p6a_{stem}.json"
    if p6_file.exists():
        p6 = mne.preprocessing.read_ica(p6_file, verbose=False)
        m = json.loads(p6_meta.read_text(encoding="utf-8"))
        excluded, power_kept = m["removed"], m["power_kept"]
    else:
        p6, excluded, power_kept = p6_ica(path)
        p6.save(p6_file, overwrite=True, verbose=False)
        p6_meta.write_text(json.dumps({"removed": excluded, "power_kept": power_kept}),
                           encoding="utf-8")
    log = json.loads((P6_LOG / f"{name}.json").read_text(encoding="utf-8"))
    p6_same = (excluded == log["removed"])
    p6_power_diff = abs(power_kept - log["power_kept"])
    print(f"{name}: P6 removed {excluded} (cache log {log['removed']}), "
          f"power kept differs by {p6_power_diff:.2e}", flush=True)

    raw = prepare(read_referential(path))
    eeg = raw.copy().pick("eeg")
    n_eeg = len(eeg.ch_names)
    icl_file = keep / f"iclabel_{stem}-ica.fif"
    if icl_file.exists():
        ica = mne.preprocessing.read_ica(icl_file, verbose=False)
    else:
        ica = mne.preprocessing.ICA(n_components=n_eeg - 1, method="infomax",
                                    fit_params=dict(extended=True),
                                    max_iter=ICA_MAX_ITER["infomax"],
                                    random_state=zlib.crc32(name.encode()) % (2 ** 32))
        ica.fit(raw, picks="eeg", decim=ICA_FIT_DECIM)
        ica.save(icl_file, overwrite=True, verbose=False)
    proba = iclabel_label_components(eeg, ica, inplace=False)
    labels = [CLASSES[int(np.argmax(p))] for p in proba]
    part = out / f"components_{name.removesuffix('.edf')}.csv"
    if part.exists():
        stored = pd.read_csv(part)
    else:                                    # runs from before the per-record checkpoints
        stored = pd.read_csv(out / "components.csv")
        stored = stored[stored["record"] == name].reset_index(drop=True)
    icl_same = labels == list(stored["iclabel"])
    print(f"{name}: ICLabel classes {'match' if icl_same else 'DIFFER from'} the stored run",
          flush=True)

    fs = eeg.info["sfreq"]
    allc = list(range(ica.n_components_))
    base = ica.apply(eeg.copy(), exclude=allc).get_data()      # PCA mean and residual
    rows = []
    for k in allc:
        only = ica.apply(eeg.copy(), exclude=[c for c in allc if c != k]).get_data() - base
        y = to_bipolar(only, eeg.ch_names)
        y = resample_poly(y, 1, int(fs // FS_CHB), axis=-1)
        y = bandpass(y * 1e6, FS_CHB, *P6_BAND).astype(np.float64) * 1e-6
        rem = p6_removed(p6, y)
        total = float((y ** 2).sum())
        rows.append({"record": name, "component": k, "iclabel": labels[k],
                     "iclabel_p": float(proba[k].max()),
                     "p6_rule_here": bool(stored.loc[k, "p6_rule"]),
                     "removed_share": float((rem ** 2).sum() / total) if total else np.nan,
                     "p6_reproduces_cache": p6_same, "p6_power_kept_diff": p6_power_diff,
                     "iclabel_reproduces": icl_same,
                     "p6_removed": " ".join(map(str, excluded))})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"p6_match_{name.removesuffix('.edf')}.csv", index=False)
    return df


def summarise(df, subject):
    lines = [f"# Share of each ICLabel component that P6 itself removes: Siena {subject}", "",
             "Generated by `python -m src.iclabel_p6_match`; do not edit by hand.", "",
             "removed_share: fraction of the component's own bipolar 0.5 to 40 Hz signal "
             "power that the recomputed P6a cleaning removes. Rows from recordings whose "
             "P6 recomputation does not reproduce the cache log are not interpretable.", ""]
    for name, d in df.groupby("record", sort=False):
        r = d.iloc[0]
        lines += [f"## {name}", "",
                  f"P6 removed components {r['p6_removed']}; reproduces the cache log: "
                  f"{r['p6_reproduces_cache']} (power kept differs by "
                  f"{r['p6_power_kept_diff']:.1e}); ICLabel run reproduces: "
                  f"{r['iclabel_reproduces']}.", "",
                  "| component | ICLabel | p | flagged by the rule here | removed by P6 |",
                  "|---|---|---|---|---|"]
        for _, x in d.sort_values("removed_share", ascending=False).head(8).iterrows():
            lines.append(f"| {x['component']} | {x['iclabel']} | {x['iclabel_p']:.2f} | "
                         f"{'yes' if x['p6_rule_here'] else 'no'} | {x['removed_share']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--records", nargs="+",
                    help="default: the recordings in which the rule flagged a component "
                         "that ICLabel calls brain (components.csv)")
    ap.add_argument("--force", action="store_true",
                    help="recompute removed_share even if p6_match_<record>.csv exists")
    args = ap.parse_args()
    out = OUT_ROOT / args.subject
    if not args.records:
        c = pd.read_csv(out / "components.csv")
        args.records = list(dict.fromkeys(c.loc[c["p6_rule"] & (c["iclabel"] == "brain"),
                                                "record"]))
        print(f"{args.subject}: records with a flagged brain component: {args.records}")
        if not args.records:
            return
    dfs = []
    for name in args.records:
        part = out / f"p6_match_{name.removesuffix('.edf')}.csv"
        dfs.append(pd.read_csv(part) if part.exists() and not args.force else match_record(args.subject, name, out))
    df = pd.concat(dfs, ignore_index=True)
    (out / "p6_match.md").write_text(summarise(df, args.subject), encoding="utf-8")
    print(f"wrote {out / 'p6_match.md'}")


if __name__ == "__main__":
    main()
