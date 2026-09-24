"""CHB-MIT üzerinde füzyon operatörü karşılaştırması, denek bazlı değerlendirme.

Bonn koşusundan üç farkı var:

  1. **Bölme denek bazlı.** Dış katmanlar leave-one-subject-out veya denek bazlı
     gruplu k-kat. İç doğrulama bölmesi de denek bazlıdır: eğitim deneklerinin bir
     kısmı tamamen ayrılır. Pencere düzeyinde iç bölme yapılsaydı aynı hastanın
     pencereleri hem eğitimde hem doğrulamada olur, erken durdurma kararı şişerdi.
  2. **Sınıf dengesizliği** 1:4 (alt örnekleme sonrası), kayıp fonksiyonu sınıf
     ağırlıklı.
  3. **Metrikler** pencere düzeyi doğruluk ve F1'in yanında olasılık tabanlı
     (Brier, log loss) ve klinik yönlü (duyarlılık, özgüllük, saatte yanlış alarm).

Kullanım:
    python -m src.chbmit_run --split loso --models late raw1d spec2d --limit-folds 2
    python -m src.chbmit_run --split loso            # tam koşu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from .chbmit_corpus import (build_corpus, grouped_kfold_by_subject,
                            leave_one_subject_out)
from .chbmit_prep import prepare
from .config import RESULTS_ROOT, fold_seed, runtime_env
from .evaluate import metrics
from .models import build, count_params
from .train import best_blend_weight, set_seed, softmax_np

OUT_ROOT = RESULTS_ROOT / "chbmit"
DEFAULT_MODELS = ("raw1d", "spec2d", "raw1d_wide", "spec2d_wide",
                  "late", "gated", "attention", "early")


def inner_split(subject: np.ndarray, tr: np.ndarray, y: np.ndarray,
                val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Eğitim katmanını denek bazlı olarak eğitim ve doğrulamaya ayırır.

    Denekler ictal sayısına göre sıralanıp dönüşümlü dağıtılır; böylece doğrulama
    kümesi ictal içermeden kalmaz. En az bir denek doğrulamaya ayrılır.
    """
    subs = sorted(set(subject[tr]))
    if len(subs) < 2:
        raise ValueError("iç doğrulama için en az iki eğitim deneği gerekir")
    counts = {s: int(y[tr][subject[tr] == s].sum()) for s in subs}
    order = sorted(subs, key=lambda s: (-counts[s], s))
    n_val = max(1, int(round(val_frac * len(subs))))
    rng = np.random.default_rng(seed)
    # ictal bakımından dengeli seçim: sıralı listeden eşit aralıklarla al
    picks = list(np.linspace(0, len(order) - 1, n_val).round().astype(int))
    val_subs = {order[i] for i in dict.fromkeys(picks)}
    if not any(counts[s] > 0 for s in val_subs):          # hiç ictal yoksa düzelt
        val_subs = {max(order, key=lambda s: counts[s])}
    rng.random()
    m = np.isin(subject[tr], list(val_subs))
    return tr[~m], tr[m]


def train_fold(model_name: str, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor,
               tr: np.ndarray, va: np.ndarray, te: np.ndarray, ncls: int, in_ch: int,
               seed: int, epochs: int = 60, min_epochs: int = 20, patience: int = 12,
               batch: int = 32, lr: float = 1e-3, wd: float = 1e-4,
               return_val_logits: bool = False,
               verbose: bool = False) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """Bir modeli bir katmanda eğitir, test olasılıklarını döndürür.

    return_val_logits=True ise üçüncü değer olarak doğrulama kümesi ham logitlerini
    (softmax öncesi) döndürür. Bu, score-level füzyon ağırlığının yalnızca doğrulama
    kümesinde seçilmesi için gerekir (bkz. src/train.py best_blend_weight).
    """
    set_seed(seed)
    model = build(model_name, ncls, in_ch=in_ch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    ytr = Y[tr].numpy()
    cnt = np.bincount(ytr, minlength=ncls).astype(np.float64)
    w = torch.tensor((cnt.sum() / (ncls * np.maximum(cnt, 1))), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)

    def raw_logits(idx: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                outs.append(model(X1[b], X2[b]).numpy())
        return np.concatenate(outs)

    def predict(idx: np.ndarray) -> np.ndarray:
        return softmax_np(raw_logits(idx))

    best_f1, best_state, bad, best_ep = -1.0, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(X1[b], X2[b]), Y[b])
            loss.backward()
            opt.step()

        pv = predict(va)
        mv = metrics(Y[va].numpy(), pv, ncls)
        if mv["f1_macro"] > best_f1:
            best_f1, best_ep = mv["f1_macro"], ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        # min_epochs koruması: model henüz tek sınıf çözümündeyken erken durdurma
        # tetiklenip çökmüş ağırlıkları "en iyi" diye saklayabiliyordu (Bonn'da
        # 55 koşunun 6'sında olmuştu).
        if ep + 1 >= min_epochs and bad >= patience:
            break
        if verbose and ep % 10 == 0:
            print(f"      ep{ep:3d} val_f1={mv['f1_macro']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    prob = predict(te)
    info = {"best_epoch": best_ep, "epochs_run": ep + 1, "val_f1": best_f1}
    if return_val_logits:
        return prob, info, raw_logits(va)
    return prob, info


def clinical_metrics(y_true: np.ndarray, prob: np.ndarray, hours: float) -> dict:
    """Duyarlılık, özgüllük ve saatte yanlış alarm.

    UYARI: saatte yanlış alarm burada **alt örneklenmiş** zaman ekseninden
    hesaplanır, yani gerçek klinik oranı temsil etmez. Gerçek oran için
    alt örneklenmemiş non-ictal pencerelerin tamamı üzerinde tahmin gerekir.
    Bu değer yalnızca modeller arası karşılaştırma içindir.
    """
    pred = prob.argmax(1)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_alarms": fp,
        "fa_per_hour_subsampled": fp / hours if hours > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["loso", "kfold"], default="loso")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit-folds", type=int, default=None,
                    help="yalnızca ilk N katmanı koş (duman testi için)")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--norm", default="window", choices=["none", "window", "channel"])
    ap.add_argument("--notch", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true",
                    help="outdir'daki perfold.csv'de tamamlanmış katmanları atla, "
                         "kaldığı yerden devam et. Bir katman yalnızca istenen "
                         "modellerin hepsi mevcutsa tamamlanmış sayılır.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t_start = time.time()

    d = build_corpus(verbose=False)
    X, y, subject = d["X"], d["y"], d["subject"]
    info = d["info"]
    ncls, in_ch = 2, X.shape[1]

    if args.norm == "channel":
        raise SystemExit("norm='channel' katman başına eğitim istatistiği gerektirir; "
                         "bu koşucuda henüz bağlanmadı, 'window' kullanın")
    x1, x2, prep_meta = prepare(X, norm=args.norm, notch_hz=args.notch)
    X1, X2, Y = torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
    del x1, x2

    splits = list(leave_one_subject_out(subject) if args.split == "loso"
                  else grouped_kfold_by_subject(subject, args.folds))
    if args.limit_folds:
        splits = splits[:args.limit_folds]

    tag = args.tag or f"{args.split}_{args.norm}" + (f"_notch{args.notch:g}" if args.notch else "")
    outdir = Path(args.outdir) if args.outdir else OUT_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "preds").mkdir(exist_ok=True)

    hours_of = {s: info["per_subject"][s]["hours"] for s in info["per_subject"]}
    rows = []
    done_folds: set[int] = set()
    csv_path = outdir / "perfold.csv"
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        have = prev.groupby("fold")["model"].apply(set)
        want = set(args.models)
        done_folds = {int(f) for f, ms in have.items() if want <= ms}
        print(f"--resume: {csv_path} okundu, {len(rows)} satır, "
              f"{len(done_folds)} katman zaten tam (atlanacak): "
              f"{sorted(done_folds)}\n")

    print(f"korpus {X.shape}, {len(splits)} katman, {len(args.models)} model, "
          f"spektrogram {X2.shape[1:]}, thread={args.threads}\n")

    for fi, (key, tr_all, te) in enumerate(splits):
        if fi in done_folds:
            print(f"[{fi+1}/{len(splits)}] atlandı (resume, zaten tam)")
            continue
        seed_f = fold_seed(20260727, 0, fi, "split")
        tr, va = inner_split(subject, tr_all, y, seed=seed_f)
        te_subs = sorted(set(subject[te]))
        hrs = sum(hours_of.get(s, 0.0) for s in te_subs)
        print(f"[{fi+1}/{len(splits)}] test={key if isinstance(key,str) else te_subs} "
              f"eğitim={len(tr)} doğrulama={len(va)} test={len(te)} "
              f"(test ictal={int(y[te].sum())})")

        probs = {}
        for m in args.models:
            t0 = time.time()
            seed = fold_seed(20260727, 0, fi, m)
            prob, tinfo = train_fold(m, X1, X2, Y, tr, va, te, ncls, in_ch, seed,
                                     epochs=args.epochs)
            mm = metrics(y[te], prob, ncls)
            mm.update(clinical_metrics(y[te], prob, hrs))
            mm.update(model=m, fold=fi, test_subjects=",".join(te_subs),
                      n_test=len(te), n_test_ictal=int(y[te].sum()),
                      params=count_params(build(m, ncls, in_ch=in_ch)),
                      sec=round(time.time() - t0, 1), **tinfo)
            rows.append(mm)
            probs[m] = prob
            print(f"    {m:12s} f1={mm['f1_macro']:.3f} auc={mm['auc']:.3f} "
                  f"brier={mm['brier']:.3f} sens={mm['sensitivity']:.3f} "
                  f"spec={mm['specificity']:.3f} ({mm['sec']:.0f}s, "
                  f"{tinfo['epochs_run']} epoch)")

        np.savez(outdir / "preds" / f"fold{fi}.npz",
                 idx_te=te, y_te=y[te], **probs)
        pd.DataFrame(rows).to_csv(outdir / "perfold.csv", index=False)

    meta = {"args": vars(args), "prep": prep_meta, "corpus": info,
            "env": runtime_env(), "elapsed_s": round(time.time() - t_start, 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    df = pd.DataFrame(rows)
    print(f"\ntoplam {meta['elapsed_s']/60:.1f} dakika, kaydedildi: {outdir}")
    print("\nmodel ortalamaları:")
    cols = ["f1_macro", "auc", "brier", "sensitivity", "specificity"]
    print(df.groupby("model")[cols].mean().round(4).sort_values("f1_macro",
                                                               ascending=False).to_string())


if __name__ == "__main__":
    main()
