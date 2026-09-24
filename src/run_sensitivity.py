"""Duyarlılık analizi: ana sonuç, tasarım kararlarımızın artefaktı mı?

Ana bulgu — "eşit bütçe altında füzyon operatörleri ayrışmıyor" — iki itiraza açıktır:

1. *"Seçtiğin spektrogram konfigürasyonu operatörleri denk yapıyor olabilir."*
2. *"Seçtiğin parametre bütçesi çok küçük; daha büyük bütçede ayrışırlar."*

Bu script her iki ekseni de tarar. Sonuç bütün noktalarda ayakta kalırsa iddia
tasarım seçimlerine bağlı değildir.

Early/intermediate füzyon dışarıda bırakılır: yapısal olarak farklı olduğu için
bütçe değiştikçe parametre eşlemesi bozulur (%16–30 sapma) ve zaten ayrı bir
performans sınıfında olduğu gösterilmiştir.

Kullanım:
    python -m src.run_sensitivity --axis stft   --variant fine
    python -m src.run_sensitivity --axis budget --variant large
    python -m src.run_sensitivity --list
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from . import baselines
from .config import RESULTS_ROOT, TASKS, Config, fold_seed
from .data import build_arrays, load_raw, select_task
from .evaluate import metrics
from .models import build, count_params
from .train import best_blend_weight, predict, softmax_np, train_one

MODELS = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide", "late", "gated", "attention"]

# Eksen -> varyant -> Config üzerine uygulanacak değişiklikler
VARIANTS = {
    "stft": {
        "fine":     dict(nfft=128, win=128, hop=32),    # zamanda ince, frekansta kaba
        "baseline": dict(nfft=256, win=256, hop=128),   # ana koşu
        "coarse":   dict(nfft=512, win=512, hop=256),   # frekansta ince, zamanda kaba
    },
    "budget": {
        "small":    dict(fusion_budget=15_000),
        "baseline": dict(fusion_budget=40_000),
        "large":    dict(fusion_budget=120_000),
    },
}


def outdir_for(axis: str, variant: str) -> "os.PathLike":
    d = RESULTS_ROOT / "sensitivity" / f"{axis}__{variant}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(cfg: Config, budget: int, axis: str, variant: str, verbose=True) -> pd.DataFrame:
    ncls = len(set(TASKS[cfg.task].values()))
    X1d, X2d, y, _ = build_arrays(cfg)
    if verbose:
        print(f"spektrogram boyutu: {X2d.shape[2]} frekans x {X2d.shape[3]} zaman", flush=True)

    X_all, sets_all = load_raw()
    X_raw, _, _ = select_task(X_all, sets_all, cfg.task)
    feats = {"logvar": baselines.logvar_features(X_raw),
             "shallow": baselines.shallow_features(X_raw)}

    t1 = torch.from_numpy(np.ascontiguousarray(X1d))
    t2 = torch.from_numpy(np.ascontiguousarray(X2d))
    ty = torch.from_numpy(y)

    rows = []
    for repeat in range(cfg.n_repeats):
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                              random_state=cfg.base_seed + repeat)
        for fold, (idx_trval, idx_te) in enumerate(skf.split(np.zeros(len(y)), y)):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_ratio,
                                         random_state=cfg.base_seed + repeat * 100 + fold)
            a, b = next(sss.split(np.zeros(len(idx_trval)), y[idx_trval]))
            idx_tr, idx_va = idx_trval[a], idx_trval[b]
            assert not (set(idx_tr) & set(idx_va) or set(idx_tr) & set(idx_te)
                        or set(idx_va) & set(idx_te)), "bölme çakışması"

            tr = (t1[idx_tr], t2[idx_tr], ty[idx_tr])
            va = (t1[idx_va], t2[idx_va], ty[idx_va])
            y_te = y[idx_te]
            uni_va, uni_te = {}, {}

            for name in MODELS:
                t0 = time.perf_counter()
                seed = fold_seed(cfg.base_seed, repeat, fold, name)
                model = train_one(name, cfg, ncls, seed, tr, va, fusion_budget=budget)
                lo_te = predict(model, t1[idx_te], t2[idx_te])
                m = metrics(y_te, softmax_np(lo_te), ncls)
                m.update(model=name, repeat=repeat, fold=fold, axis=axis, variant=variant,
                         params=count_params(model), sec=time.perf_counter() - t0)
                rows.append(m)
                if name in ("raw1d", "spec2d"):
                    uni_va[name] = predict(model, t1[idx_va], t2[idx_va])
                    uni_te[name] = lo_te
                if verbose:
                    print(f"  r{repeat} f{fold} {name:12s} F1={m['f1_macro']:.4f} "
                          f"({m['sec']:.1f}s)", flush=True)

            w = best_blend_weight(uni_va["raw1d"], uni_va["spec2d"], y[idx_va], ncls)
            p = w * softmax_np(uni_te["raw1d"]) + (1 - w) * softmax_np(uni_te["spec2d"])
            m = metrics(y_te, p, ncls)
            m.update(model="score", repeat=repeat, fold=fold, axis=axis, variant=variant,
                     params=np.nan, sec=0.0)
            rows.append(m)

            for name, feat in feats.items():
                prob = baselines.fit_predict(feat[idx_tr], y[idx_tr], feat[idx_te])
                m = metrics(y_te, prob, ncls)
                m.update(model=name, repeat=repeat, fold=fold, axis=axis, variant=variant,
                         params=feat.shape[1] * ncls, sec=0.0)
                rows.append(m)

        pd.DataFrame(rows).to_csv(outdir_for(axis, variant) / "perfold.csv", index=False)
        if verbose:
            print(f"[repeat {repeat} bitti] {axis}/{variant}", flush=True)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(VARIANTS))
    ap.add_argument("--variant")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_THREADS", "2")))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for ax, vs in VARIANTS.items():
            for v, kw in vs.items():
                print(f"  --axis {ax} --variant {v:9s} {kw}")
        return

    torch.set_num_threads(args.threads)
    kw = dict(VARIANTS[args.axis][args.variant])
    budget = kw.pop("fusion_budget", 40_000)
    cfg = replace(Config(task="T1_3class", norm_mode="N2_z_zspec",
                         n_repeats=args.repeats), **kw)

    print(f"eksen={args.axis} varyant={args.variant} butce={budget:,} "
          f"stft(nfft={cfg.nfft},win={cfg.win},hop={cfg.hop}) tekrar={cfg.n_repeats}")
    df = run(cfg, budget, args.axis, args.variant)

    s = (df.groupby("model").agg(f1=("f1_macro", "mean"), sd=("f1_macro", "std"),
                                 params=("params", "max"))
           .sort_values("f1", ascending=False))
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
