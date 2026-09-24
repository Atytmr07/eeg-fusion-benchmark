"""CHB-MIT korpus düzeyinde veri kurma ve sızıntıya kapalı bölme.

Bu modülün tek amacı, Bonn üzerinde yapılamayan şeyi yapmak: **denek bazlı**
değerlendirme. Bonn denek kimliği yayımlamadığı için orada bir modelin yeni bir
hastaya genelleyip genellemediği sorulamıyordu.

Sızıntıya karşı iki kural:

  1. Aynı **deneğin** hiçbir penceresi hem eğitimde hem testte olamaz.
  2. Aynı **nöbetin** hiçbir penceresi bölünemez. Örtüşmeli pencerelemede aynı
     nöbetin pencereleri neredeyse aynıdır; bölünürlerse model ezberi genelleme
     sanır. Kural 1 uygulandığında bu kendiliğinden sağlanır, ama örtüşmeli
     pencereleme veya denek içi bölme denenirse ayrıca gerekir. Bu yüzden nöbet
     kimlikleri veriyle birlikte taşınır ve `check_leakage` ikisini de denetler.

Yükleyici sessiz boş küme döndürmez: beklenen denek, kayıt ve kanal sayısı
doğrulanır, uyuşmazlıkta hata verilir. Gerekçe, bu projede iki kez yaşanan hata
sınıfıdır (chb17'nin dosya adı kalıbı yüzünden tamamen atlanması, chb12'nin farklı
montajı yüzünden kanal kesişiminin boş çıkması); ikisi de hata vermeden sıfır
döndürüyordu.

Kullanım:
    python -m src.chbmit_corpus --build          # önbelleği kur
    python -m src.chbmit_corpus --check          # sızıntı ve bütünlük testi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit import (CHB_ROOT, FS_CHB, TARGET_CHANNELS, extract_windows,
                     parse_summary, plan_windows, seizure_ids, subject_files)

CACHE = CHB_ROOT / "_cache"
NON_ICTAL = ""                      # ictal olmayan pencerelerin nöbet kimliği


def all_subjects() -> list[str]:
    return sorted(d.name for d in CHB_ROOT.glob("chb*") if d.is_dir())


def _window_seizure_id(t0: float, win_s: float,
                       zs: list[tuple[float, float, str]]) -> str:
    """Pencerenin en çok örtüştüğü nöbetin kimliği. Örtüşme yoksa boş dize."""
    best, best_ov = NON_ICTAL, 0.0
    for b0, b1, sid in zs:
        ov = max(0.0, min(t0 + win_s, b1) - max(t0, b0))
        if ov > best_ov:
            best, best_ov = sid, ov
    return best


def build_corpus(subjects: list[str] | None = None, win_s: float = 10.0,
                 stride_s: float | None = None, neg_per_pos: float = 4.0,
                 guard_s: float = 0.0, seed: int = 20260727,
                 cache: bool = True, verbose: bool = True) -> dict:
    """Korpusu pencereler, alt örnekler ve grup etiketleriyle birlikte döndürür.

    Dönen sözlük: X, y, subject, seizure_id, record, t0, info.

    Alt örnekleme **denek içinde** yapılır: her deneğin kendi ictal pencere sayısına
    oranla non-ictal pencere tutulur. Korpus genelinde tek bir oran kullanmak,
    nöbeti çok olan deneklerin non-ictal örneklerini diğerlerinin üstüne yığardı.
    """
    subs = subjects or all_subjects()
    tag = (f"w{win_s:g}_s{(stride_s or win_s):g}_n{neg_per_pos:g}"
           f"_g{guard_s:g}_seed{seed}_{len(subs)}subj")
    npz = CACHE / f"corpus_{tag}.npz"
    if cache and npz.exists():
        if verbose:
            print(f"önbellekten okunuyor: {npz.name}")
        d = np.load(npz, allow_pickle=False)
        info = json.loads((CACHE / f"corpus_{tag}.json").read_text(encoding="utf-8"))
        return {k: d[k] for k in d.files} | {"info": info}

    rng = np.random.default_rng(seed)
    Xs, ys, subj, sids, fnames, t0s = [], [], [], [], [], []
    per_subject = {}

    for si, sub in enumerate(subs):
        files_all = subject_files(sub, require_channels=False)
        files = subject_files(sub)
        if not files:
            raise RuntimeError(f"{sub}: hedef montajı içeren kayıt yok "
                               f"({len(files_all)} kayıt tarandı)")
        summ = parse_summary(CHB_ROOT / sub / f"{sub}-summary.txt")
        zmap = seizure_ids(sub)

        # 1. geçiş: başlıklardan etiket ve grup planı, sinyal okunmadan
        plan = []                                    # (dosya, t0, y, seizure_id)
        for f in files:
            y_f, t_f = plan_windows(f, summ.get(f.name, []), win_s, stride_s, guard_s)
            zs = zmap.get(f.name, [])
            for yy, tt in zip(y_f, t_f):
                sid = _window_seizure_id(float(tt), win_s, zs) if yy == 1 else NON_ICTAL
                plan.append((f.name, float(tt), int(yy), sid))
        if not plan:
            raise RuntimeError(f"{sub}: pencere üretilemedi")

        y_all = np.array([p[2] for p in plan])
        pos = np.flatnonzero(y_all == 1)
        neg = np.flatnonzero(y_all == 0)
        if len(pos) == 0:
            if verbose:
                print(f"{sub}: ictal pencere yok, atlanıyor")
            continue
        keep_neg = neg
        if neg_per_pos > 0 and len(neg) > neg_per_pos * len(pos):
            keep_neg = rng.choice(neg, int(neg_per_pos * len(pos)), replace=False)
        idx = np.sort(np.concatenate([pos, keep_neg]))

        # 2. geçiş: yalnızca seçilen pencereler, dosya başına tek okuma
        by_file: dict[str, list[tuple[int, float]]] = {}
        for j, i in enumerate(idx):
            by_file.setdefault(plan[i][0], []).append((j, plan[i][1]))
        X = np.empty((len(idx), len(TARGET_CHANNELS), int(win_s * FS_CHB)), np.float32)
        path_of = {f.name: f for f in files}
        for name, items in by_file.items():
            js = np.array([a for a, _ in items])
            ts = np.array([b for _, b in items], np.float32)
            X[js] = extract_windows(path_of[name], list(TARGET_CHANNELS), ts, win_s)

        Xs.append(X)
        ys.append(y_all[idx])
        subj += [sub] * len(idx)
        sids += [plan[i][3] for i in idx]
        fnames += [plan[i][0] for i in idx]
        t0s += [plan[i][1] for i in idx]
        per_subject[sub] = {
            "n_files": len(files), "n_files_skipped_montage": len(files_all) - len(files),
            "n_windows_total": int(len(plan)), "n_ictal": int(len(pos)),
            "n_nonictal_kept": int(len(keep_neg)),
            "n_seizures": int(len({p[3] for p in plan if p[3]})),
            "hours": float(len(plan) * (stride_s or win_s) / 3600.0),
        }
        if verbose:
            print(f"[{si+1:2d}/{len(subs)}] {sub}: {len(idx):5d} pencere "
                  f"({len(pos):4d} ictal, {per_subject[sub]['n_seizures']:2d} nöbet, "
                  f"{per_subject[sub]['hours']:6.1f} saat)")

    out = {
        "X": np.concatenate(Xs),
        "y": np.concatenate(ys).astype(np.int64),
        "subject": np.array(subj),
        "seizure_id": np.array(sids),
        "record": np.array(fnames),
        "t0": np.array(t0s, np.float32),
    }
    info = {
        "subjects": subs, "n_subjects": len(per_subject),
        "channels": list(TARGET_CHANNELS), "fs": FS_CHB,
        "win_s": win_s, "stride_s": stride_s or win_s, "guard_s": guard_s,
        "neg_per_pos": neg_per_pos, "seed": seed,
        "n_windows": int(len(out["y"])), "n_ictal": int(out["y"].sum()),
        "n_seizures": int(len({s for s in sids if s})),
        "per_subject": per_subject,
    }
    validate(out, info)

    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **out)
        (CACHE / f"corpus_{tag}.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"önbelleğe yazıldı: {npz} ({npz.stat().st_size/1e6:.0f} MB)")
    return out | {"info": info}


def validate(d: dict, info: dict) -> None:
    """Sessiz boş küme ve biçim bozukluklarına karşı kontroller. Hata fırlatır."""
    n = len(d["y"])
    if n == 0:
        raise RuntimeError("korpus boş")
    for k in ("X", "subject", "seizure_id", "record", "t0"):
        if len(d[k]) != n:
            raise RuntimeError(f"{k} uzunluğu {len(d[k])}, y uzunluğu {n}")
    if d["X"].shape[1] != len(TARGET_CHANNELS):
        raise RuntimeError(f"kanal sayısı {d['X'].shape[1]}, "
                           f"beklenen {len(TARGET_CHANNELS)}")
    if d["X"].shape[2] != int(info["win_s"] * FS_CHB):
        raise RuntimeError(f"pencere uzunluğu {d['X'].shape[2]}, "
                           f"beklenen {int(info['win_s'] * FS_CHB)}")
    if not np.isfinite(d["X"]).all():
        bad = int((~np.isfinite(d["X"])).sum())
        raise RuntimeError(f"sinyalde {bad} sonlu olmayan değer var")
    if d["y"].sum() == 0:
        raise RuntimeError("hiç ictal pencere yok")
    # her ictal pencerenin bir nöbet kimliği olmalı, non-ictal olanın olmamalı
    ict = d["y"] == 1
    if (d["seizure_id"][ict] == NON_ICTAL).any():
        raise RuntimeError("nöbet kimliği olmayan ictal pencere var")
    if (d["seizure_id"][~ict] != NON_ICTAL).any():
        raise RuntimeError("nöbet kimliği taşıyan non-ictal pencere var")
    if len(set(d["subject"])) < 2:
        print("uyarı: korpusta ikiden az denek var, denek bazlı bölme yapılamaz")


# --- Bölme ------------------------------------------------------------------

def leave_one_subject_out(subject: np.ndarray):
    """Her denek sırayla test kümesi olur. Denek bazlı genellemenin ölçüsü budur."""
    for s in sorted(set(subject)):
        te = np.flatnonzero(subject == s)
        tr = np.flatnonzero(subject != s)
        yield s, tr, te


def grouped_kfold_by_subject(subject: np.ndarray, n_folds: int = 5,
                             seed: int = 20260727):
    """Denekleri katmanlara dağıtır. LOSO çok pahalıysa kullanılır.

    Denekler ictal pencere sayısına göre sıralanıp sırayla en az yüklü katmana
    atanır; böylece katmanlar arasında ictal sayısı dengelenir ve hiçbir katman
    ictal içermeden kalmaz.
    """
    subs = sorted(set(subject))
    if n_folds > len(subs):
        raise ValueError(
            f"{n_folds} katman isteniyor ama {len(subs)} denek var; en az bir "
            f"katman boş kalır. Katman sayısını düşürün veya "
            f"leave_one_subject_out kullanın.")
    rng = np.random.default_rng(seed)
    counts = {s: int((subject == s).sum()) for s in subs}
    order = sorted(subs, key=lambda s: (-counts[s], s))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    load = [0] * n_folds
    for s in order:
        k = int(np.argmin(load))
        folds[k].append(s)
        load[k] += counts[s]
    for k in range(n_folds):
        rng.shuffle(folds[k])
        te_subs = set(folds[k])
        te = np.flatnonzero(np.isin(subject, list(te_subs)))
        tr = np.flatnonzero(~np.isin(subject, list(te_subs)))
        yield sorted(te_subs), tr, te


def check_leakage(d: dict, splitter, name: str = "") -> list[str]:
    """Bölmelerin iki kuralı da sağladığını denetler. Sorun listesi döndürür."""
    problems = []
    subject, sid, y = d["subject"], d["seizure_id"], d["y"]
    for key, tr, te in splitter:
        if len(tr) == 0 or len(te) == 0:
            problems.append(f"{name} {key}: boş katman (eğitim {len(tr)}, test {len(te)})")
            continue
        shared_s = set(subject[tr]) & set(subject[te])
        if shared_s:
            problems.append(f"{name} {key}: denek hem eğitimde hem testte: "
                            f"{sorted(shared_s)}")
        a = {s for s in sid[tr] if s != NON_ICTAL}
        b = {s for s in sid[te] if s != NON_ICTAL}
        if a & b:
            problems.append(f"{name} {key}: nöbet bölünmüş: {sorted(a & b)[:5]}")
        if y[te].sum() == 0:
            problems.append(f"{name} {key}: test katmanında ictal pencere yok")
        if y[tr].sum() == 0:
            problems.append(f"{name} {key}: eğitim katmanında ictal pencere yok")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--guard", type=float, default=0.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = build_corpus(args.subjects, win_s=args.win, stride_s=args.stride,
                     neg_per_pos=args.neg_per_pos, guard_s=args.guard)
    info = d["info"]
    print(f"\nkorpus: {info['n_windows']} pencere, {info['n_ictal']} ictal, "
          f"{info['n_seizures']} nöbet, {info['n_subjects']} denek, "
          f"{d['X'].shape[1]} kanal x {d['X'].shape[2]} örnek")
    print(f"bellek: {d['X'].nbytes/1e9:.2f} GB")

    if args.check:
        n_subj = len(set(d["subject"]))
        folds = min(args.folds, n_subj)
        if folds != args.folds:
            print(f"\nnot: {args.folds} katman istendi ama {n_subj} denek var, "
                  f"{folds} katmana düşürüldü")

        print("\n--- sızıntı denetimi ---")
        p1 = check_leakage(d, leave_one_subject_out(d["subject"]), "LOSO")
        p2 = check_leakage(d, grouped_kfold_by_subject(d["subject"], folds),
                           f"{folds}-kat")
        for p in p1 + p2:
            print("  SORUN:", p)
        if not (p1 or p2):
            print(f"  LOSO {n_subj} katman ve {folds} katlı gruplu bölme: "
                  f"sızıntı yok")

        print("\n--- katman dengesi (gruplu bölme) ---")
        for te_subs, tr, te in grouped_kfold_by_subject(d["subject"], folds):
            print(f"  test denekleri {len(te_subs):2d}: {len(te):5d} pencere, "
                  f"{int(d['y'][te].sum()):4d} ictal  |  eğitim {len(tr):5d}")

        print("\n--- yapay sızıntı testi (denetimin çalıştığının kanıtı) ---")
        def bad_splitter():
            """Kasten sızdıran bölme: pencereler rastgele ikiye ayrılır."""
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(d["y"]))
            half = len(idx) // 2
            yield "rastgele", idx[:half], idx[half:]
        caught = check_leakage(d, bad_splitter(), "YAPAY")
        if caught:
            print(f"  denetim yapay sızıntıyı yakaladı ({len(caught)} bulgu):")
            for c in caught[:3]:
                print("   ", c[:110])
        else:
            print("  UYARI: denetim yapay sızıntıyı YAKALAYAMADI, denetim bozuk")


if __name__ == "__main__":
    main()
