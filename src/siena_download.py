"""Download the Siena Scalp EEG database and audit its EDF headers remotely.

Files come from PhysioNet's open-data mirror on S3, which serves the same files as
physionet.org and supports HTTP Range requests. That makes two things possible:

  - an interrupted download resumes where it stopped, and a file whose size already
    matches the server's is skipped; every completed EDF is checked against the
    SHA-256 published in SHA256SUMS.txt
  - the header audit (--audit) reads only the EDF headers, 256 bytes plus 256 per
    signal, about 0.5 MB for all 41 recordings instead of 20.3 GB

Usage:
    python -m src.siena_download --audit                    # header audit only
    python -m src.siena_download --subjects PN00            # one subject (~400 MB)
    python -m src.siena_download                            # everything (20.3 GB)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import read_edf_header
from .siena import (CORRECTIONS, ELECTRODES, FS_SIENA, SIENA_ROOT, SUBJECTS, clock,
                    electrode_index, parse_seizure_list, physical_dimensions,
                    recording_start)

BASE_URL = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/"
META_FILES = ("RECORDS", "subject_info.csv", "SHA256SUMS.txt", "LICENSE.txt")
CHUNK = 1 << 20


def _get(url: str, start: int = 0, end: int | None = None):
    headers = {}
    if start or end is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                  timeout=60)


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(rel: str, root: Path = SIENA_ROOT, verbose: bool = True) -> Path:
    """Download one file to root/rel, resuming a partial .part file. A file whose size
    already equals the server's is skipped."""
    url, dest = BASE_URL + rel, root / rel
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size == size:
        if verbose:
            print(f"  {rel}: present ({size/1e6:.0f} MB), skipped")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        with _get(url, start=have) as r, open(part, "ab") as fh:
            done, last = have, -1
            for block in iter(lambda: r.read(CHUNK), b""):
                fh.write(block)
                done += len(block)
                pct = int(100 * done / size)
                if verbose and pct // 10 != last:
                    last = pct // 10
                    print(f"\r  {rel}: {done/1e6:7.0f} / {size/1e6:.0f} MB", end="",
                          flush=True)
        if verbose:
            print()
    if part.stat().st_size != size:
        raise RuntimeError(f"{rel}: {part.stat().st_size} bytes received, "
                           f"{size} expected; run again to resume")
    part.replace(dest)
    return dest


def download(subjects: list[str] | None = None, root: Path = SIENA_ROOT) -> None:
    subs = subjects or list(SUBJECTS)
    unknown = sorted(set(subs) - set(SUBJECTS))
    if unknown:
        raise SystemExit(f"unknown subjects: {unknown}; known: {' '.join(SUBJECTS)}")
    print("metadata:")
    for f in META_FILES:
        fetch(f, root)
    sums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    records = (root / "RECORDS").read_text().split()

    for sub in subs:
        print(f"{sub}:")
        fetch(f"{sub}/Seizures-list-{sub}.txt", root)
        for rec in [r for r in records if r.startswith(sub + "/")]:
            path = fetch(rec, root)
            want = sums.get(rec)
            if want is None:
                print(f"  {rec}: no published checksum")
            elif sha256(path) != want:
                path.rename(path.with_name(path.name + ".bad"))
                raise RuntimeError(f"{rec}: SHA-256 mismatch; file renamed to .bad, "
                                   f"run again to download it afresh")
    print("done; check with: python -m src.verify_siena")


# --- Header audit ----------------------------------------------------------------------------

def remote_header(rel: str) -> Path:
    """The EDF header of a remote file, saved to a temporary file so the project's own
    header reader can parse it."""
    url = BASE_URL + rel
    with _get(url, 0, 255) as r:
        first = r.read()
        total = int(r.headers["Content-Range"].split("/")[1])
    ns = int(first[252:256].decode("ascii").strip())
    with _get(url, 256, 256 + 256 * ns - 1) as r:
        rest = r.read()
    tmp = Path(tempfile.mkdtemp()) / Path(rel).name
    tmp.write_bytes(first + rest)
    tmp.with_suffix(".size").write_text(str(total))
    return tmp


def audit_record(header: Path, file_size: int, listed: list) -> dict:
    """Compare one recording's EDF header with its seizure list entries."""
    h = read_edf_header(header)
    start = recording_start(header)
    row = {"file": header.name, "header_start": clock(start),
           "duration_s": h.duration, "header_end": clock(start + h.duration),
           "fs": sorted(set(h.fs)), "n_signals": len(h.labels),
           "size_ok": file_size == h.header_bytes + h.n_records * sum(h.samples_per_record) * 2}
    try:
        idx = electrode_index(h.labels, header.name)
        dims = physical_dimensions(header)
        row["electrodes"] = "all 19"
        row["units"] = sorted({dims[i] for i in idx.values()})
        row["o1_label"] = h.labels[idx["O1"]]
    except ValueError as e:
        row["electrodes"] = str(e)
    mine = [z for z in listed if z.file == header.name]
    if mine:
        z = mine[0]
        row["list_start"], row["list_end"] = clock(z.reg_start), clock(z.reg_end)
        d0 = (z.reg_start - start + 43200) % 86400 - 43200      # signed, seconds
        d1 = (z.reg_end - (start + h.duration) + 43200) % 86400 - 43200
        row["start_diff_s"], row["end_diff_s"] = int(d0), int(round(d1))
        row["n_seizures"] = len(mine)
    else:
        row["n_seizures"] = 0
    return row


def audit(root: Path = SIENA_ROOT) -> list[dict]:
    """Header audit of all recordings without downloading the signals."""
    for f in ("RECORDS",):
        fetch(f, root, verbose=False)
    records = (root / "RECORDS").read_text().split()
    rows = []
    for sub in SUBJECTS:
        fetch(f"{sub}/Seizures-list-{sub}.txt", root, verbose=False)
        listed = parse_seizure_list(sub, root)
        for rec in [r for r in records if r.startswith(sub + "/")]:
            hdr = remote_header(rec)
            size = int(hdr.with_suffix(".size").read_text())
            rows.append({"subject": sub} | audit_record(hdr, size, listed))

    print(f"{'file':20s} {'header start':>12s} {'list start':>10s} {'diff':>6s}   "
          f"{'header end':>10s} {'list end':>8s} {'diff':>6s}  seizures")
    for r in rows:
        flag = "" if r.get("start_diff_s") == 0 else "  <-- start differs"
        print(f"{r['file']:20s} {r['header_start']:>12s} {r.get('list_start', '-'):>10s} "
              f"{r.get('start_diff_s', '-'):>6}   {r['header_end']:>10s} "
              f"{r.get('list_end', '-'):>8s} {r.get('end_diff_s', '-'):>6}  "
              f"{r['n_seizures']}{flag}")

    print("\nsummary:")
    print(f"  recordings: {len(rows)}, with listed seizures: "
          f"{sum(r['n_seizures'] > 0 for r in rows)}")
    print(f"  sampling rates: {sorted({f for r in rows for f in r['fs']})} Hz "
          f"(expected {FS_SIENA:g})")
    print(f"  file size equals header-computed size: "
          f"{sum(r['size_ok'] for r in rows)}/{len(rows)}")
    print(f"  all {len(ELECTRODES)} electrodes present: "
          f"{sum(r['electrodes'] == 'all 19' for r in rows)}/{len(rows)}")
    print(f"  electrode units: {sorted({u for r in rows for u in r.get('units', [])})}")
    print(f"  O1 label in the headers: {sorted({r.get('o1_label') for r in rows})} "
          f"(the lists say '1')")
    starts = [r for r in rows if r.get("start_diff_s") not in (0, None)]
    print(f"  list start == header start: {len(rows) - len(starts)}/{len(rows)}")
    for r in starts:
        known = CORRECTIONS["registration_start"].get((r["subject"], r["file"]))
        note = "documented in CORRECTIONS" if known == r["header_start"] else "NOT DOCUMENTED"
        print(f"    {r['file']}: list {r['list_start']}, header {r['header_start']} "
              f"({r['start_diff_s']:+d} s), {note}")
    ends = [r for r in rows if r.get("end_diff_s") not in (0, None)]
    print(f"  list end == header start + duration: {len(rows) - len(ends)}/{len(rows)}")
    for r in ends:
        print(f"    {r['file']}: list {r['list_end']}, header {r['header_end']} "
              f"({r['end_diff_s']:+d} s)")

    out = root / "header_audit.csv"
    keys = ["subject", "file", "header_start", "list_start", "start_diff_s",
            "header_end", "list_end", "end_diff_s", "duration_s", "fs", "n_signals",
            "size_ok", "electrodes", "units", "o1_label", "n_seizures"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subjects to download (default: all 14)")
    ap.add_argument("--audit", action="store_true",
                    help="header audit only, downloads about 0.5 MB")
    args = ap.parse_args()
    if args.audit:
        audit()
    else:
        download(args.subjects)


if __name__ == "__main__":
    main()
