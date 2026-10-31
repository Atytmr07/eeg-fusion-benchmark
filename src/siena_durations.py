"""Recording duration of each Siena patient, measured three ways.

  header  sum of the EDF headers (n_records x record duration), from header_audit.csv
  list    sum of "Registration end - start" in the seizure lists, modulo 24 hours,
          one term per file (merged files such as PN10-4.5.6 listed once)
  info    rec_time_minutes in subject_info.csv; the database description's "about 128
          recording hours" is this column's total

Needs only data/siena/header_audit.csv (python -m src.siena_download --audit, about
0.5 MB) and subject_info.csv, not the signals.

Usage:   python -m src.siena_durations
Output:  console table + data/siena/durations.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from .siena import SIENA_ROOT


def _seconds(clock: str) -> int:
    h, m, s = (int(x) for x in clock.split("."))
    return h * 3600 + m * 60 + s


def durations(root: Path = SIENA_ROOT) -> list[dict]:
    audit = list(csv.DictReader(open(root / "header_audit.csv", encoding="utf-8")))
    info = {r["patient_id"]: r for r in
            csv.DictReader(open(root / "subject_info.csv", encoding="utf-8"),
                           skipinitialspace=True)}
    out: dict[str, dict] = {}
    for r in audit:
        d = out.setdefault(r["subject"], {"subject": r["subject"], "files": 0,
                                          "header_min": 0.0, "list_min": 0.0})
        d["files"] += 1
        d["header_min"] += float(r["duration_s"]) / 60
        d["list_min"] += ((_seconds(r["list_end"]) - _seconds(r["list_start"])) % 86400) / 60
    for sub, d in out.items():
        d["info_min"] = float(info[sub]["rec_time_minutes"])
        d["info_minus_header_min"] = d["info_min"] - d["header_min"]
        d["info_minus_list_min"] = d["info_min"] - d["list_min"]
    return list(out.values())


def main() -> None:
    rows = durations()
    print(f"{'subject':8}{'files':>6}{'header':>10}{'list':>10}{'info':>10}"
          f"{'info-hdr':>10}{'info-list':>10}   (minutes)")
    for d in rows:
        print(f"{d['subject']:8}{d['files']:6d}{d['header_min']:10.1f}{d['list_min']:10.1f}"
              f"{d['info_min']:10.0f}{d['info_minus_header_min']:10.1f}"
              f"{d['info_minus_list_min']:10.1f}")
    tot = {k: sum(d[k] for d in rows) for k in ("header_min", "list_min", "info_min")}
    print(f"{'total':8}{sum(d['files'] for d in rows):6d}{tot['header_min']:10.1f}"
          f"{tot['list_min']:10.1f}{tot['info_min']:10.0f}")
    print(f"hours: header {tot['header_min'] / 60:.2f}, list {tot['list_min'] / 60:.2f}, "
          f"info {tot['info_min'] / 60:.2f}")
    path = SIENA_ROOT / "durations.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for d in rows:
            w.writerow({k: round(v, 1) if isinstance(v, float) else v for k, v in d.items()})
    print(f"written: {path}")


if __name__ == "__main__":
    main()
