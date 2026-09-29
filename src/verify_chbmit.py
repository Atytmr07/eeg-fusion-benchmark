"""Integrity and consistency check of a CHB-MIT download.

Why header checks instead of a list of file sizes: every EDF header states its number
of records, channels and samples per record, from which the exact file size follows.
So no external manifest of 686 files is needed, and a truncated download cannot pass
silently.

Checks:
  1. each EDF's size equals the size computed from its header
  2. sampling rate is 256 Hz and uniform across channels
  3. each case has a summary file and it parses
  4. files with seizures in the summary match the .seizures files present
  5. seizure intervals lie inside the recording and start < end

Usage:   python -m src.verify_chbmit
Output:  console report + data/chbmit/verification.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import CHB_ROOT, parse_summary, read_edf_header


def expected_size(h) -> int:
    return h.header_bytes + h.n_records * sum(h.samples_per_record) * 2


def check_subject(sub: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    edfs = sorted(sub.glob("*.edf"))
    summ_path = sub / f"{sub.name}-summary.txt"

    seiz_files = {p.name[:-len(".seizures")] for p in sub.glob("*.edf.seizures")}
    summ: dict = {}
    if not summ_path.exists():
        problems.append(f"{sub.name}: summary file missing")
    else:
        try:
            summ = parse_summary(summ_path)
        except Exception as e:
            problems.append(f"{sub.name}: summary could not be parsed: {e}")

    with_seiz = {k for k, v in summ.items() if v}
    if summ and seiz_files and with_seiz != seiz_files:
        only_summ = sorted(with_seiz - seiz_files)
        only_file = sorted(seiz_files - with_seiz)
        problems.append(f"{sub.name}: summary and .seizures files disagree "
                        f"(only in summary: {only_summ}, only as file: {only_file})")

    for f in edfs:
        row = {"subject": sub.name, "file": f.name, "ok": True, "note": ""}
        try:
            h = read_edf_header(f)
        except Exception as e:
            row.update(ok=False, note=f"header unreadable: {e}")
            problems.append(f"{f.name}: header unreadable")
            rows.append(row)
            continue

        size, want = f.stat().st_size, expected_size(h)
        row.update(n_channels=len(h.labels), duration_s=h.duration,
                   fs=h.fs[0] if h.fs else None, size=size, expected=want)
        if size < want:
            row.update(ok=False, note=f"truncated: size {size}, expected {want}")
            problems.append(f"{f.name}: truncated (size {size} < {want})")
        elif size > want:
            # chb17b_69.edf carries 256 bytes of zero padding at the source; the
            # server's Content-Length matches, so this is not a bad download. The
            # reader ignores the excess. Only non-zero excess is reported.
            with open(f, "rb") as fh:
                fh.seek(want)
                tail = fh.read(size - want)
            if any(tail):
                row.update(ok=False, note=f"{size - want} extra bytes, not zero")
                problems.append(f"{f.name}: {size - want} bytes more than expected "
                                f"and the padding is not zero")
            else:
                row["note"] = f"{size - want} bytes of zero padding (as at the source)"
        if len(set(h.samples_per_record)) != 1:
            row.update(ok=False, note=(row["note"] + " mixed sampling rates").strip())
            problems.append(f"{f.name}: channels have different sampling rates")
        elif abs(h.fs[0] - 256.0) > 1e-6:
            row.update(ok=False, note=(row["note"] + f" fs={h.fs[0]}").strip())
            problems.append(f"{f.name}: sampling rate is not 256 Hz ({h.fs[0]})")

        for s in summ.get(f.name, []):
            if not (0 <= s.start_s < s.end_s <= h.duration):
                row.update(ok=False, note=(row["note"] + " invalid seizure interval").strip())
                problems.append(f"{f.name}: seizure interval {s.start_s}-{s.end_s} "
                                f"inconsistent with recording length {h.duration}")
        rows.append(row)
    return rows, problems


def main() -> None:
    subs = sorted(d for d in CHB_ROOT.glob("chb*") if d.is_dir())
    if not subs:
        print(f"no case folders in {CHB_ROOT}")
        return

    all_rows, all_problems = [], []
    for sub in subs:
        rows, problems = check_subject(sub)
        all_rows += rows
        all_problems += problems
        n_ok = sum(r["ok"] for r in rows)
        hours = sum(r.get("duration_s", 0) for r in rows) / 3600
        summ_path = sub / f"{sub.name}-summary.txt"
        n_seiz = sum(len(v) for v in parse_summary(summ_path).values()) \
            if summ_path.exists() else 0
        flag = "" if n_ok == len(rows) else "  <-- PROBLEM"
        print(f"{sub.name}: {n_ok}/{len(rows)} files intact, {hours:6.1f} hours, "
              f"{n_seiz:3d} seizures{flag}")

    out = CHB_ROOT / "verification.csv"
    if all_rows:
        keys = ["subject", "file", "ok", "n_channels", "duration_s", "fs",
                "size", "expected", "note"]
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

    total_h = sum(r.get("duration_s", 0) for r in all_rows) / 3600
    n_ok = sum(r["ok"] for r in all_rows)
    print("=" * 62)
    print(f"cases={len(subs)}  files={len(all_rows)}  intact={n_ok}  "
          f"total={total_h:.1f} hours")
    if all_problems:
        print(f"\n{len(all_problems)} problems:")
        for p in all_problems[:40]:
            print("  -", p)
        if len(all_problems) > 40:
            print(f"  ... and {len(all_problems) - 40} more")
    else:
        print("no problems")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
