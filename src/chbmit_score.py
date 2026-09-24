"""CHB-MIT LOSO'ya score-level füzyonu ekler.

`chbmit_run.py`'nin ana koşusu 8 derin modeli kapsıyordu ama `score` (raw1d ve
spec2d'nin doğrulama kümesinde seçilen ağırlıkla harmanlanması) dahil değildi.
Bunu eklemek raw1d ve spec2d'yi yeniden eğitmeyi gerektiriyor, çünkü asıl koşu
doğrulama kümesi olasılıklarını saklamamıştı (yalnızca test tahminleri kaydedildi).

`train_fold`'u `chbmit_run.py`'den olduğu gibi kullanır, aynı tohumlama ve aynı
bölmelerle. Bu yüzden buradan çıkan raw1d/spec2d test performansı, ana koşudaki
raw1d/spec2d satırlarıyla birebir aynı olmalı (aynı seed, aynı veri); bu betik
sonunda bunu doğrular.

Kullanım:  python -m src.chbmit_score --threads 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_corpus import build_corpus, leave_one_subject_out
from .chbmit_prep import prepare
from .chbmit_run import clinical_metrics, inner_split, train_fold
from .config import RESULTS_ROOT, fold_seed
from .evaluate import metrics
from .models import build, count_params
from .train import best_blend_weight

OUT_DIR = RESULTS_ROOT / "chbmit" / "loso_main"
CSV_PATH = OUT_DIR / "perfold.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--limit-folds", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = build_corpus(verbose=False)
    X, y, subject = d["X"], d["y"], d["subject"]
    info = d["info"]
    ncls, in_ch = 2, X.shape[1]
    hours_of = {s: info["per_subject"][s]["hours"] for s in info["per_subject"]}

    x1, x2, _ = prepare(X, norm="window")
    X1, X2, Y = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
    del x1, x2

    splits = list(leave_one_subject_out(subject))
    if args.limit_folds:
        splits = splits[:args.limit_folds]

    done_folds: set[int] = set()
    old_df = None
    if CSV_PATH.exists():
        old_df = pd.read_csv(CSV_PATH)
        if args.resume and "score" in set(old_df.model):
            done_folds = set(old_df[old_df.model == "score"].fold.astype(int))
            print(f"--resume: {len(done_folds)} katmanda score zaten var, "
                  f"atlanacak: {sorted(done_folds)}\n")

    check_rows = []      # raw1d/spec2d karsilastirmasi icin
    new_rows = []
    print(f"score füzyonu için {len(splits)} katman, thread={args.threads}\n")

    for fi, (key, tr_all, te) in enumerate(splits):
        if fi in done_folds:
            print(f"[{fi+1}/{len(splits)}] atlandı (resume)")
            continue
        seed_f = fold_seed(20260727, 0, fi, "split")
        tr, va = inner_split(subject, tr_all, y, seed=seed_f)
        te_subs = sorted(set(subject[te]))
        hrs = sum(hours_of.get(s, 0.0) for s in te_subs)

        val_logits, test_prob = {}, {}
        for m in ("raw1d", "spec2d"):
            t0 = time.time()
            seed = fold_seed(20260727, 0, fi, m)
            prob, tinfo, vl = train_fold(m, X1, X2, Y, tr, va, te, ncls, in_ch, seed,
                                         epochs=args.epochs, return_val_logits=True)
            test_prob[m] = prob
            val_logits[m] = vl
            mm = metrics(y[te], prob, ncls)
            check_rows.append({"fold": fi, "model": m, "f1_macro": mm["f1_macro"],
                               "sec": round(time.time() - t0, 1)})
            print(f"    {m:8s} f1={mm['f1_macro']:.3f} ({time.time()-t0:.0f}s, "
                  f"{tinfo['epochs_run']} epoch)")

        w = best_blend_weight(val_logits["raw1d"], val_logits["spec2d"],
                              Y[va].numpy(), ncls)
        p_score = w * test_prob["raw1d"] + (1 - w) * test_prob["spec2d"]
        mm = metrics(y[te], p_score, ncls)
        mm.update(clinical_metrics(y[te], p_score, hrs))
        mm.update(model="score", fold=fi, test_subjects=",".join(te_subs),
                  n_test=len(te), n_test_ictal=int(y[te].sum()), blend_w=w,
                  params=count_params(build("raw1d", ncls, in_ch=in_ch)) +
                         count_params(build("spec2d", ncls, in_ch=in_ch)),
                  sec=0.0, best_epoch=-1, epochs_run=0, val_f1=float("nan"))
        new_rows.append(mm)
        print(f"[{fi+1}/{len(splits)}] {te_subs}: score f1={mm['f1_macro']:.3f} "
              f"blend_w={w:.2f}\n")

    if not new_rows:
        print("eklenecek yeni satır yok")
        return

    # raw1d/spec2d'nin bu yeniden koşusu ana koşudakiyle uyuşuyor mu doğrula
    if old_df is not None:
        chk = pd.DataFrame(check_rows)
        ref = old_df[old_df.model.isin(("raw1d", "spec2d"))]
        merged = chk.merge(ref, on=["fold", "model"], suffixes=("_new", "_orig"))
        diff = (merged.f1_macro_new - merged.f1_macro_orig).abs()
        print(f"raw1d/spec2d tutarlılık kontrolü ({len(merged)} satır): "
              f"maks fark={diff.max():.4f}  ortalama fark={diff.mean():.4f}")
        if diff.max() > 0.01:
            print("UYARI: fark 0.01'i aşıyor, thread sayısı farkından olabilir "
                  "(bkz. docs/10). score sonucu yine de kaydediliyor.")

    new_df = pd.DataFrame(new_rows)
    if old_df is not None:
        backup = OUT_DIR / "perfold_before_score.csv"
        if not backup.exists():
            import shutil
            shutil.copy(CSV_PATH, backup)
            print(f"yedeklendi: {backup}")
        kept = old_df[~((old_df.model == "score") &
                        (old_df.fold.isin(new_df.fold.unique())))]
        merged_out = pd.concat([kept, new_df], ignore_index=True)
    else:
        merged_out = new_df

    merged_out.to_csv(CSV_PATH, index=False)
    print(f"\nkaydedildi: {CSV_PATH} ({len(merged_out)} satır)")
    print(new_df[["fold", "f1_macro", "auc", "brier", "blend_w"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
