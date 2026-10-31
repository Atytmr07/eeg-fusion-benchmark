"""Integrity and consistency check of a Siena download.

Like src/verify_chbmit.py, sizes are checked against the EDF headers rather than a
file list, so a truncated download cannot pass silently. Only the subjects present on
disk are checked; recordings listed in RECORDS but absent are reported as missing.

Checks:
  1. each EDF's size equals the size computed from its header
  2. sampling rate is 512 Hz and uniform across channels
  3. all 19 electrodes needed for the 18 bipolar channels are present, in microvolts
  4. the seizure list parses (with CORRECTIONS and DECISIONS)
  5. every seizure lies inside its own recording, relative to the EDF start time
  6. the list's registration start time agrees with the EDF header (disagreements
     are reported; those documented in CORRECTIONS are not counted as problems)

Usage:   python -m src.verify_siena
Output:  console report + data/siena/verification.csv
"""
from __future__ import annotations

import csv
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import read_edf_header
from .siena import (CORRECTIONS, FS_SIENA, SIENA_ROOT, SUBJECTS, clock,
                    electrode_index, parse_seizure_list, physical_dimensions,
                    record_path, recording_start, seizures_in_record, subject_records)
from .verify_chbmit import expected_size


def check_subject(sub: str) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    try:
        listed = parse_seizure_list(sub)
    except Exception as e:
        return [], [f"{sub}: seizure list could not be parsed: {e}"]

    for name in subject_records(sub):
        f = record_path(sub, name)
        row = {"subject": sub, "file": name, "ok": True, "note": ""}
        if not f.exists():
            row.update(ok=False, note="missing")
            problems.append(f"{name}: missing")
            rows.append(row)
            continue
        try:
            h = read_edf_header(f)
        except Exception as e:
            row.update(ok=False, note=f"header unreadable: {e}")
            problems.append(f"{name}: header unreadable")
            rows.append(row)
            continue

        def fail(msg: str) -> None:
            row.update(ok=False, note=(row["note"] + " " + msg).strip())
            problems.append(f"{name}: {msg}")

        size, want = f.stat().st_size, expected_size(h)
        start = recording_start(f)
        row.update(n_signals=len(h.labels), duration_s=h.duration,
                   fs=h.fs[0] if h.fs else None, size=size, expected=want,
                   start=clock(start))
        if size != want:
            fail(f"size {size}, expected {want} from the header")
        if len(set(h.samples_per_record)) != 1:
            fail("channels have different sampling rates")
        elif abs(h.fs[0] - FS_SIENA) > 1e-6:
            fail(f"sampling rate {h.fs[0]}, expected {FS_SIENA:g}")
        try:
            idx = electrode_index(h.labels, name)
            dims = physical_dimensions(f)
            units = {dims[i] for i in idx.values()}
            if units != {"uV"}:
                fail(f"electrode units {sorted(units)}, expected uV")
        except ValueError as e:
            fail(str(e))

        mine = [z for z in listed if z.file == name]
        if not mine:
            fail("no seizure in the list (every Siena recording has one)")
        else:
            if mine[0].reg_start != start:
                known = CORRECTIONS["registration_start"].get((sub, name))
                msg = (f"list start {clock(mine[0].reg_start)}, header start "
                       f"{clock(start)}")
                if known == clock(start):
                    row["note"] = (row["note"] + f" {msg} (documented)").strip()
                else:
                    fail(msg + " (not in CORRECTIONS)")
            try:
                zs, excl, _ = seizures_in_record(f, listed)
                row["n_seizures"] = len(zs)
                row["n_excluded"] = len(excl)
                row["seizure_s"] = sum(z.duration for z in zs)
            except ValueError as e:
                fail(str(e))
        rows.append(row)
    return rows, problems


def main() -> None:
    if not (SIENA_ROOT / "RECORDS").exists():
        print(f"no RECORDS in {SIENA_ROOT}; download with python -m src.siena_download")
        return
    subs = [s for s in SUBJECTS
            if any(record_path(s, r).exists() for r in subject_records(s))]
    if not subs:
        print(f"no recordings in {SIENA_ROOT}")
        return

    all_rows, all_problems = [], []
    for sub in subs:
        rows, problems = check_subject(sub)
        all_rows += rows
        all_problems += problems
        n_ok = sum(r["ok"] for r in rows)
        hours = sum(r.get("duration_s", 0) for r in rows) / 3600
        n_seiz = sum(r.get("n_seizures", 0) for r in rows)
        secs = sum(r.get("seizure_s", 0) for r in rows)
        flag = "" if n_ok == len(rows) and rows else "  <-- PROBLEM"
        print(f"{sub}: {n_ok}/{len(rows)} files intact, {hours:6.2f} hours, "
              f"{n_seiz:2d} seizures, {secs:5.0f} s of seizure{flag}")

    out = SIENA_ROOT / "verification.csv"
    keys = ["subject", "file", "ok", "n_signals", "duration_s", "fs", "size",
            "expected", "start", "n_seizures", "n_excluded", "seizure_s", "note"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    total_h = sum(r.get("duration_s", 0) for r in all_rows) / 3600
    n_ok = sum(r["ok"] for r in all_rows)
    print("=" * 62)
    print(f"subjects={len(subs)}/{len(SUBJECTS)}  files={len(all_rows)}  intact={n_ok}  "
          f"total={total_h:.1f} hours")
    notes = [f"{r['file']}: {r['note']}" for r in all_rows if r["ok"] and r["note"]]
    for nline in notes:
        print("  note:", nline)
    if all_problems:
        print(f"\n{len(all_problems)} problems:")
        for p in all_problems[:40]:
            print("  -", p)
    else:
        print("no problems")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
