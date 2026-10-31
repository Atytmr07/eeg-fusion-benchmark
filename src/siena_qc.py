"""Quality checks behind the Siena seizure-list decisions (docs/SIENA.md, Section 5).

Two checks, both reproducible from the raw recordings:

  PN00 seizure 3   The list gives its end as 19.29.29, after the recording ends. The
                   signal decides between the 18.29.29 reading and excluding the seizure:
                   the right frontotemporal ictal rhythm of seizure 3 is compared with the
                   patient's other four seizures, and a figure is written to
                   docs/figures/siena_PN00_sz3.png.
  PN10 onsets      For every PN10 seizure, which onset the list gives (unmarked, clinical,
                   electrical) and how many 10 s window labels would change between the
                   clinical and the electrical onset.

Usage:
    python -m src.siena_qc --pn00          # needs data/siena/PN00
    python -m src.siena_qc --pn10          # seizure lists only, plus PN10 headers if present
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import FS_CHB, TARGET_CHANNELS, Seizure, plan_windows
from .config import PROJECT_ROOT
from .siena import (DECISIONS, SIENA_ROOT, _TIME, _seconds, clock, parse_seizure_list,
                    read_bipolar, record_path, recording_start)

FIG_DIR = PROJECT_ROOT / "docs" / "figures"

# PN00's seizures are right temporal (subject_info.csv); its ictal rhythm is clearest on
# the right frontotemporal chain.
FOCUS = ("FP2-F8", "F8-T8")
RHYTHM_BAND = (3.0, 20.0)   # Hz: above post-ictal delta, below most muscle activity
OFFSET_FRAC = 0.25          # offset: band power below this share of its ictal peak ...
OFFSET_HOLD = 10            # ... for this many consecutive seconds
SMOOTH_S = 5


def _band_power_1s(x: np.ndarray, fs: float, chans: tuple[str, ...]) -> np.ndarray:
    """Power in RHYTHM_BAND per 1 s step, summed over `chans`, smoothed over SMOOTH_S."""
    from scipy.signal import spectrogram
    total = 0.0
    for c in chans:
        f, _, p = spectrogram(x[TARGET_CHANNELS.index(c)], fs, nperseg=int(fs),
                              noverlap=0)
        total = total + p[(f >= RHYTHM_BAND[0]) & (f <= RHYTHM_BAND[1])].sum(axis=0)
    return np.convolve(total, np.ones(SMOOTH_S) / SMOOTH_S, mode="same")


def ictal_offset(power: np.ndarray, onset_s: int, search_s: int = 120) -> tuple[int, int]:
    """(peak, offset) in seconds: the band-power peak within `search_s` of the listed
    onset, and the first second after it from which the power stays below OFFSET_FRAC
    of the peak for OFFSET_HOLD seconds."""
    peak = onset_s + int(np.argmax(power[onset_s:onset_s + search_s]))
    thr = OFFSET_FRAC * power[peak]
    for t in range(peak, len(power) - OFFSET_HOLD):
        if (power[t:t + OFFSET_HOLD] < thr).all():
            return peak, t
    return peak, len(power)


def pn00_check(make_figure: bool = True) -> list[dict]:
    """Compare the ictal rhythm of PN00 seizure 3 with the other four seizures."""
    from .preprocess import bandpass

    decisions = dict(DECISIONS, PN00_seizure3="typo")
    listed = parse_seizure_list("PN00", decisions=decisions)
    rows, signals = [], {}
    for z in listed:
        path = record_path("PN00", z.file)
        x = read_bipolar(path)
        t0 = recording_start(path)
        on, end = (z.start - t0) % 86400, (z.end - t0) % 86400
        pw = _band_power_1s(x, FS_CHB, FOCUS)
        peak, off = ictal_offset(pw, on)
        rows.append({"seizure": z.number, "file": z.file, "onset": clock(z.start),
                     "listed_end": clock(z.end), "listed_s": end - on,
                     "peak_s": peak - on, "offset_s": off - on,
                     "offset_minus_listed_end": off - end})
        signals[z.number] = (x, t0, on, end, pw)

    print("Right frontotemporal (FP2-F8, F8-T8) 3-20 Hz power; offset = below "
          f"{OFFSET_FRAC:.0%} of the ictal peak for {OFFSET_HOLD} s")
    print(f"  {'seizure':8s} {'file':12s} {'onset':>8s} {'listed end':>10s} "
          f"{'listed s':>8s} {'peak +s':>7s} {'offset +s':>9s} {'offset - end':>12s}")
    for r in rows:
        tag = "  <- 18.29.29 reading" if r["seizure"] == 3 else ""
        print(f"  {r['seizure']:8d} {r['file']:12s} {r['onset']:>8s} {r['listed_end']:>10s} "
              f"{r['listed_s']:8d} {r['peak_s']:7d} {r['offset_s']:9d} "
              f"{r['offset_minus_listed_end']:+12d}{tag}")
    others = [r["offset_minus_listed_end"] for r in rows if r["seizure"] != 3]
    sz3 = next(r for r in rows if r["seizure"] == 3)
    x3, t03, on3, _, _ = signals[3]
    rec_end = t03 + x3.shape[1] / FS_CHB
    print(f"\n  other seizures: signal offset {min(others):+d} to {max(others):+d} s from "
          f"the listed end; seizure 3 with the 18.29.29 reading: "
          f"{sz3['offset_minus_listed_end']:+d} s")
    print(f"  the listed 19.29.29 lies {(_seconds('19.29.29') - rec_end):.0f} s after the "
          f"end of the recording ({clock(rec_end)})")

    if make_figure:
        _pn00_figure(signals, rows, bandpass)
    return rows


def _pn00_figure(signals: dict, rows: list[dict], bandpass) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = FS_CHB
    x, t0, on, end, pw = signals[3]
    a, b = _seconds("18.27.30") - t0, _seconds("18.31.00") - t0
    seg = bandpass(x[:, int(a * fs):int(b * fs)], fs, 0.5, 40.0)
    t = a + np.arange(seg.shape[1]) / fs

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.2, 1.3], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    # one common scale and clipping keep each channel within its own row, so relative
    # amplitudes stay comparable across channels
    scale = 2.5 * np.percentile(np.abs(seg), 95)
    for i in range(len(TARGET_CHANNELS)):
        ax.plot(t, np.clip(seg[i] / scale, -0.5, 0.5) - i, lw=0.35, color="0.15")
    ax.set_yticks(-np.arange(len(TARGET_CHANNELS)))
    ax.set_yticklabels(TARGET_CHANNELS, fontsize=8)
    ax.set_ylim(-len(TARGET_CHANNELS) + 0.3, 1.9)
    marks = [(on, "listed onset 18:28:29", "tab:green"),
             (end, "corrected end 18:29:29", "tab:red")]
    for s, lab, col in marks:
        ax.axvline(s, color=col, lw=1.4)
        ax.text(s + 1, 0.75, lab, color=col, fontsize=9, va="bottom")
    ax.text(b - 1, 1.2, "listed end 19:29:29: after the end of the recording "
            "(18:57:33) →", ha="right", va="bottom", fontsize=8, color="tab:red")
    ticks = np.arange(np.ceil(a / 30) * 30, b + 1, 30)
    ax.set_xticks(ticks)
    ax.set_xticklabels([clock(t0 + v)[:8].replace(".", ":") for v in ticks])
    ax.set_xlim(a, b)
    ax.set_title("PN00 seizure 3 (PN00-3.edf), 18 bipolar channels, 0.5-40 Hz", fontsize=11)

    # per-channel line length, 1 s steps, relative to the pre-ictal median
    ax2 = fig.add_subplot(gs[1])
    n = x.shape[1] // int(fs)
    ll = np.abs(np.diff(x[:, :n * int(fs)].reshape(len(TARGET_CHANNELS), n, int(fs)),
                        axis=-1)).sum(-1)
    base = np.median(ll[:, max(0, on - 300):on - 30], axis=1, keepdims=True)
    ia, ib = int(a), int(b)
    im = ax2.imshow(np.log2(ll[:, ia:ib] / base), aspect="auto", cmap="magma",
                    extent=(ia, ib, len(TARGET_CHANNELS) - 0.5, -0.5), vmin=-1, vmax=3)
    for s, _, col in marks:
        ax2.axvline(s, color=col, lw=1.2)
    ax2.set_yticks(range(0, len(TARGET_CHANNELS), 3))
    ax2.set_yticklabels(TARGET_CHANNELS[::3], fontsize=7)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([clock(t0 + v)[:8].replace(".", ":") for v in ticks])
    ax2.set_title("Line length per channel, 1 s steps, log2 ratio to the pre-ictal median",
                  fontsize=10)
    fig.colorbar(im, ax=ax2, pad=0.01, fraction=0.02)

    # 3-20 Hz power of all five seizures, aligned to their listed onsets
    ax3 = fig.add_subplot(gs[2])
    for r in rows:
        xi, _, oi, ei, pi = signals[r["seizure"]]
        rel = np.arange(-60, 181)
        ok = (oi + rel >= 0) & (oi + rel < len(pi))
        y = pi[oi + rel[ok]] / pi[oi:oi + 120].max()
        others = {1: "#4c72b0", 2: "#55a868", 4: "#8172b2", 5: "#64b5cd"}
        style = (dict(color="#c44e52", lw=2.2) if r["seizure"] == 3 else
                 dict(color=others[r["seizure"]], lw=1.0, alpha=0.85))
        line, = ax3.plot(rel[ok], y, label=f"seizure {r['seizure']} "
                         f"(listed {r['listed_s']} s, offset {r['offset_s']} s)", **style)
        ax3.plot([ei - oi], [OFFSET_FRAC], marker="v", color=line.get_color(), ms=8)
    ax3.axhline(OFFSET_FRAC, color="0.5", lw=0.8, ls="--")
    ax3.axvline(0, color="tab:green", lw=1.0)
    ax3.set_xlabel("seconds from listed onset")
    ax3.set_ylabel("3-20 Hz power / peak")
    ax3.set_title("Right frontotemporal (FP2-F8, F8-T8) 3-20 Hz power, all five PN00 "
                  "seizures (▼ listed end)", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(-60, 180)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "siena_PN00_sz3.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out}")
    return out


def pn10_onsets() -> list[dict]:
    """Clinical versus electrical onset of every PN10 seizure and its effect on labels."""
    text = (SIENA_ROOT / "PN10" / "Seizures-list-PN10.txt").read_text(encoding="latin-1")
    raw_onsets: dict[int, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"\s*Seizure n\s*(\d+)", line, re.I)
        if m:
            current = int(m.group(1))
        m = re.match(r"\s*Seizure start time:\s*(.*)", line, re.I)
        if m and current is not None:
            raw_onsets[current] = m.group(1).strip()

    elec = parse_seizure_list("PN10", decisions=dict(DECISIONS, PN10_seizure3_onset="electric"))
    clin = parse_seizure_list("PN10", decisions=dict(DECISIONS, PN10_seizure3_onset="clinical"))
    rows = []
    for ze, zc in zip(elec, clin):
        raw = raw_onsets[ze.number]
        marks = re.findall(r"\(([^)]*)\)", raw)
        kind = ("both" if len(_TIME.findall(raw)) > 1 else
                "clinical only" if any("CLINICAL" in m.upper() for m in marks) else
                "electrical only" if any("ELECTRIC" in m.upper() for m in marks) else
                "unmarked")
        row = {"seizure": ze.number, "file": ze.file, "listed": raw, "kind": kind,
               "clinical": clock(zc.start) if kind in ("both", "clinical only") else "",
               "electrical": clock(ze.start) if kind in ("both", "electrical only") else "",
               "difference_s": ze.start - zc.start if kind == "both" else None}
        path = record_path("PN10", ze.file)
        if kind == "both" and path.exists():
            t0 = recording_start(path)
            labels = {}
            for name, z in (("clinical", zc), ("electrical", ze)):
                s, e = (z.start - t0) % 86400, (z.end - t0) % 86400
                for guard in (0.0, 1.0):
                    y, t = plan_windows(path, [Seizure(path.name, float(s), float(e))],
                                        guard_s=guard)
                    labels[(name, guard)] = dict(zip(t.round(3).tolist(), y.tolist()))
            a, b = labels[("clinical", 0.0)], labels[("electrical", 0.0)]
            row["windows_changed"] = sum(a[k] != b[k] for k in a)
            a, b = labels[("clinical", 1.0)], labels[("electrical", 1.0)]
            row["windows_changed_boundary_dropped"] = (
                len(set(a) ^ set(b)) + sum(a[k] != b[k] for k in set(a) & set(b)))
        elif kind == "both":
            # without the EDF: window starts lie on a 10 s grid from the header start
            # PN10-3 header start time, from the header audit (docs/SIENA.md, Section 3)
            t0 = _seconds("13.33.18")
            s_c, s_e = (zc.start - t0) % 86400, (ze.start - t0) % 86400
            row["windows_changed"] = int(np.floor(s_e / 10) != np.floor(s_c / 10))
        rows.append(row)

    print("PN10 onsets (10 s windows, boundary windows labelled ictal as in the main runs)")
    print(f"  {'sz':>3s} {'file':16s} {'listed onset':52s} {'kind':16s} {'diff s':>6s} "
          f"{'windows changed':>15s}")
    for r in rows:
        d = "" if r["difference_s"] is None else f"{r['difference_s']:+d}"
        w = r.get("windows_changed", "")
        print(f"  {r['seizure']:3d} {r['file']:16s} {r['listed'][:52]:52s} {r['kind']:16s} "
              f"{d:>6s} {w!s:>15s}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pn00", action="store_true", help="PN00 seizure 3 check and figure")
    ap.add_argument("--pn10", action="store_true", help="PN10 onset table")
    args = ap.parse_args()
    if not (args.pn00 or args.pn10):
        args.pn00 = args.pn10 = True
    if args.pn00:
        pn00_check()
    if args.pn10:
        if args.pn00:
            print()
        pn10_onsets()


if __name__ == "__main__":
    main()
