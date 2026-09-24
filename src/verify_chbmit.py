"""CHB-MIT indirmesinin bütünlüğünü ve iç tutarlılığını doğrular.

Neden boyut listesi değil de başlık kontrolü: EDF başlığı kayıt sayısını, kanal
sayısını ve kayıt başına örnek sayısını kendisi taşır. Beklenen dosya boyutu bu
üçünden hesaplanır. Böylece 686 dosya için dışarıdan bir manifest tutmaya gerek
kalmaz ve kesik indirme sessizce geçemez.

Kontroller:
  1. Her EDF'in dosya boyutu, başlıktan hesaplanan beklenen boyuta eşit mi
  2. Örnekleme hızı 256 Hz mi, kanallar tek tip hızda mı
  3. Her deneğin özet dosyası var mı, ayrıştırılabiliyor mu
  4. Özetteki nöbetli kayıt sayısı, .seizures dosya sayısıyla tutuyor mu
  5. Nöbet aralıkları kayıt süresinin içinde mi ve başlangıç < bitiş mi

Kullanım:  python -m src.verify_chbmit
Çıktı:     konsol raporu + data/chbmit/verification.csv
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
        problems.append(f"{sub.name}: özet dosyası yok")
    else:
        try:
            summ = parse_summary(summ_path)
        except Exception as e:
            problems.append(f"{sub.name}: özet ayrıştırılamadı: {e}")

    with_seiz = {k for k, v in summ.items() if v}
    if summ and seiz_files and with_seiz != seiz_files:
        only_summ = sorted(with_seiz - seiz_files)
        only_file = sorted(seiz_files - with_seiz)
        problems.append(f"{sub.name}: özet ile .seizures uyuşmuyor "
                        f"(yalnız özette: {only_summ}, yalnız dosyada: {only_file})")

    for f in edfs:
        row = {"subject": sub.name, "file": f.name, "ok": True, "note": ""}
        try:
            h = read_edf_header(f)
        except Exception as e:
            row.update(ok=False, note=f"başlık okunamadı: {e}")
            problems.append(f"{f.name}: başlık okunamadı")
            rows.append(row)
            continue

        size, want = f.stat().st_size, expected_size(h)
        row.update(n_channels=len(h.labels), duration_s=h.duration,
                   fs=h.fs[0] if h.fs else None, size=size, expected=want)
        if size < want:
            row.update(ok=False, note=f"kesik: boyut {size} beklenen {want}")
            problems.append(f"{f.name}: kesik (boyut {size} < {want})")
        elif size > want:
            # chb17b_69.edf kaynakta 256 bayt sıfır dolgu taşıyor. Sunucudaki
            # Content-Length de aynı olduğu için bu eksik indirme değil, kaynağın
            # özelliği. Okuyucu fazlalığı zaten yok sayar. Yalnızca fazlalık
            # sıfırdan farklıysa sorun bildirilir.
            with open(f, "rb") as fh:
                fh.seek(want)
                tail = fh.read(size - want)
            if any(tail):
                row.update(ok=False, note=f"fazla veri {size - want} bayt, sıfır değil")
                problems.append(f"{f.name}: beklenenden {size - want} bayt fazla "
                                f"ve dolgu sıfır değil")
            else:
                row["note"] = f"{size - want} bayt sıfır dolgu (kaynakta böyle)"
        if len(set(h.samples_per_record)) != 1:
            row.update(ok=False, note=(row["note"] + " karışık örnekleme").strip())
            problems.append(f"{f.name}: kanallar farklı örnekleme hızında")
        elif abs(h.fs[0] - 256.0) > 1e-6:
            row.update(ok=False, note=(row["note"] + f" fs={h.fs[0]}").strip())
            problems.append(f"{f.name}: örnekleme 256 Hz değil ({h.fs[0]})")

        for s in summ.get(f.name, []):
            if not (0 <= s.start_s < s.end_s <= h.duration):
                row.update(ok=False, note=(row["note"] + " nöbet aralığı geçersiz").strip())
                problems.append(f"{f.name}: nöbet aralığı {s.start_s}-{s.end_s} "
                                f"kayıt süresi {h.duration} ile bağdaşmıyor")
        rows.append(row)
    return rows, problems


def main() -> None:
    subs = sorted(d for d in CHB_ROOT.glob("chb*") if d.is_dir())
    if not subs:
        print(f"denek klasörü yok: {CHB_ROOT}")
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
        flag = "" if n_ok == len(rows) else "  <-- SORUN"
        print(f"{sub.name}: {n_ok}/{len(rows)} dosya sağlam, {hours:6.1f} saat, "
              f"{n_seiz:3d} nöbet{flag}")

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
    print(f"denek={len(subs)}  dosya={len(all_rows)}  sağlam={n_ok}  "
          f"toplam={total_h:.1f} saat")
    if all_problems:
        print(f"\n{len(all_problems)} sorun:")
        for p in all_problems[:40]:
            print("  -", p)
        if len(all_problems) > 40:
            print(f"  ... ve {len(all_problems) - 40} tane daha")
    else:
        print("sorun yok")
    print(f"kaydedildi: {out}")


if __name__ == "__main__":
    main()
