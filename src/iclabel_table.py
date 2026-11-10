"""Table 1 of the ICLabel technical report: components removed by each rule, per patient.

Rule A: the P6 frontopolar rule (p6_rule). Rule B: ICLabel class eye. Rule C: keep only
ICLabel class brain (so it removes every other class). "Most ictal": components in the top
fifth of the 30 to 100 Hz seizure/background power ratio within their recording.

Usage: .venv-ica/Scripts/python -m src.iclabel_table
Output: results_v2/iclabel/table1.csv, table1.md
"""
from __future__ import annotations

import pandas as pd

from .iclabel_check import OUT_ROOT
from .siena import parse_seizure_list, record_path, seizures_in_record

PATIENTS = ("PN00", "PN12", "PN14", "PN10", "PN17")


def main():
    rows = []
    for s in PATIENTS:
        c = pd.read_csv(OUT_ROOT / s / "components.csv")
        eye = c["iclabel"] == "eye blink"
        a, rule_c = c["p6_rule"], c["iclabel"] != "brain"
        top = c.groupby("record")["ictal_gamma_ratio"].transform(lambda g: g >= g.quantile(0.8))
        seizures = sum(len(seizures_in_record(record_path(s, r), parse_seizure_list(s))[0])
                       for r in c["record"].unique())
        rows.append({"patient": s, "recordings": c["record"].nunique(), "seizures": seizures,
                     "components": len(c), "rule_A": int(a.sum()), "rule_B": int(eye.sum()),
                     "both": int((a & eye).sum()), "rule_C": int(rule_c.sum()),
                     "rule_C_pct": round(100 * rule_c.mean()),
                     "most_ictal": int(top.sum()), "most_ictal_A": int((top & a).sum()),
                     "most_ictal_B": int((top & eye).sum()),
                     "most_ictal_C": int((top & rule_c).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "table1.csv", index=False)
    lines = ["| Patient | Recordings / seizures | Components | Rule A | Rule B | Both | Rule C "
             "| Most ictal removed, A / B / C |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['patient']} | {r['recordings']} / {r['seizures']} | {r['components']} "
                     f"| {r['rule_A']} | {r['rule_B']} | {r['both']} | {r['rule_C']} "
                     f"({r['rule_C_pct']}%) | {r['most_ictal_A']} / {r['most_ictal_B']} / "
                     f"{r['most_ictal_C']} of {r['most_ictal']} |")
    (OUT_ROOT / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
