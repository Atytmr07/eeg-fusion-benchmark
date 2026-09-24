"""Ana deney: tekrarlı katmanlı çapraz doğrulama ile füzyon operatörü karşılaştırması.

Kullanım:
    python -m src.run_benchmark --task T1_3class --norm N2_z_zspec
    python -m src.run_benchmark --task T1_3class --norm N2_z_zspec --repeats 1 --folds 2  # smoke

Çıktı: <outdir>/perfold.csv  — her (tekrar, fold, model) için bir satır.
Sığ referans modeller (logvar, shallow) derin modellerle AYNI bölmeler üzerinde koşar,
böylece eşleştirilmiş istatistiksel teste birlikte girebilirler.
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
from .config import CLASS_NAMES, Config, NORM_MODES, TASKS, fold_seed
from .data import build_arrays, load_raw, select_task
from .evaluate import metrics
from .models import build, count_params
from .train import best_blend_weight, predict, softmax_np, train_one

DEEP_MODELS = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide",
               "late", "gated", "attention", "early"]
SHALLOW_MODELS = ["logvar", "shallow"]


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def run(cfg: Config, verbose: bool = True, outdir=None) -> pd.DataFrame:
    # Thread sayısı sonuçları değiştirdiği için konfigürasyondan okunur; çağıran
    # kim olursa olsun aynı değer yürürlükte olsun diye burada set edilir.
    torch.set_num_threads(cfg.threads)
    ncls = len(set(TASKS[cfg.task].values()))
    X1d, X2d, y, sets = build_arrays(cfg)

    # Sığ baseline'lar için ham sinyal (normalizasyondan bağımsız)
    X_all, sets_all = load_raw()
    X_raw, y_raw, _ = select_task(X_all, sets_all, cfg.task)
    assert np.array_equal(y_raw, y)
    feat_logvar = baselines.logvar_features(X_raw)
    feat_shallow = baselines.shallow_features(X_raw)

    t1, t2, ty = _t(X1d), _t(X2d), _t(y)
    rows = []
    from pathlib import Path
    outdir = Path(outdir) if outdir else cfg.outdir()
    cfg.save(outdir)
    pred_dir = outdir / "preds"
    pred_dir.mkdir(exist_ok=True)

    for repeat in range(cfg.n_repeats):
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                              random_state=cfg.base_seed + repeat)
        for fold, (idx_trval, idx_te) in enumerate(skf.split(np.zeros(len(y)), y)):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_ratio,
                                         random_state=cfg.base_seed + repeat * 100 + fold)
            a, b = next(sss.split(np.zeros(len(idx_trval)), y[idx_trval]))
            idx_tr, idx_va = idx_trval[a], idx_trval[b]

            # sızıntı kontrolü (assert — sessizce geçilmez)
            assert not (set(idx_tr) & set(idx_va)), "train/val çakışması"
            assert not (set(idx_tr) & set(idx_te)), "train/test çakışması"
            assert not (set(idx_va) & set(idx_te)), "val/test çakışması"

            tr = (t1[idx_tr], t2[idx_tr], ty[idx_tr])
            va = (t1[idx_va], t2[idx_va], ty[idx_va])
            y_te = y[idx_te]

            uni_logits_va, uni_logits_te = {}, {}
            fold_probs = {}

            for name in DEEP_MODELS:
                t0 = time.perf_counter()
                seed = fold_seed(cfg.base_seed, repeat, fold, name)
                model = train_one(name, cfg, ncls, seed, tr, va)
                logit_te = predict(model, t1[idx_te], t2[idx_te])
                prob_te = softmax_np(logit_te)
                fold_probs[name] = prob_te.astype(np.float32)
                m = metrics(y_te, prob_te, ncls)
                m.update(model=name, repeat=repeat, fold=fold,
                         params=count_params(model), sec=time.perf_counter() - t0)
                rows.append(m)
                if name in ("raw1d", "spec2d"):
                    uni_logits_va[name] = predict(model, t1[idx_va], t2[idx_va])
                    uni_logits_te[name] = logit_te
                if verbose:
                    print(f"  r{repeat} f{fold} {name:12s} F1={m['f1_macro']:.4f} "
                          f"AUC={m['auc']:.4f} ({m['sec']:.1f}s)", flush=True)

            # score-level füzyon: ağırlık YALNIZCA validation üzerinde seçilir
            t0 = time.perf_counter()
            w = best_blend_weight(uni_logits_va["raw1d"], uni_logits_va["spec2d"],
                                  y[idx_va], ncls)
            p = w * softmax_np(uni_logits_te["raw1d"]) + \
                (1 - w) * softmax_np(uni_logits_te["spec2d"])
            fold_probs["score"] = p.astype(np.float32)
            m = metrics(y_te, p, ncls)
            m.update(model="score", repeat=repeat, fold=fold, blend_w=w,
                     params=count_params(build("raw1d", ncls)) +
                            count_params(build("spec2d", ncls)),
                     sec=time.perf_counter() - t0)
            rows.append(m)

            # sığ referanslar — aynı bölme
            for name, feat in (("logvar", feat_logvar), ("shallow", feat_shallow)):
                t0 = time.perf_counter()
                prob = baselines.fit_predict(feat[idx_trval], y[idx_trval], feat[idx_te])
                fold_probs[name] = prob.astype(np.float32)
                m = metrics(y_te, prob, ncls)
                m.update(model=name, repeat=repeat, fold=fold,
                         params=feat.shape[1] * ncls, sec=time.perf_counter() - t0)
                rows.append(m)
                if verbose:
                    print(f"  r{repeat} f{fold} {name:12s} F1={m['f1_macro']:.4f} "
                          f"AUC={m['auc']:.4f}", flush=True)

            np.savez_compressed(pred_dir / f"r{repeat}_f{fold}.npz",
                                idx_te=idx_te, y_te=y_te, **fold_probs)

        df = pd.DataFrame(rows)
        df.to_csv(outdir / "perfold.csv", index=False)
        if verbose:
            print(f"[repeat {repeat} bitti] -> {outdir/'perfold.csv'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "perfold.csv", index=False)

    summary = (df.groupby("model")
                 .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                      auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                      acc_mean=("acc", "mean"),
                      recall_ictal_mean=("recall_ictal", "mean"),
                      params=("params", "max"), sec_mean=("sec", "mean"), n=("f1_macro", "size"))
                 .sort_values("f1_mean", ascending=False))
    summary.to_csv(outdir / "summary.csv")
    if verbose:
        print(summary.round(4).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1_3class", choices=list(TASKS))
    ap.add_argument("--norm", default="N2_z_zspec", choices=list(NORM_MODES))
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lowpass", action="store_true")
    ap.add_argument("--outdir", default=None,
                    help="Varsayılan config-hash dizini yerine buraya yaz")
    ap.add_argument("--threads", type=int, default=None,
                    help="Varsayılan Config.threads yerine kullanılacak thread sayısı. "
                         "Sonuçları değiştirir ve koşu hash'ine dahildir.")
    args = ap.parse_args()

    cfg = Config(task=args.task, norm_mode=args.norm, apply_lowpass=args.lowpass)
    over = {}
    threads = args.threads
    if threads is None and os.environ.get("TORCH_THREADS"):
        threads = int(os.environ["TORCH_THREADS"])
    if threads is not None:
        over["threads"] = threads
    if args.repeats is not None:
        over["n_repeats"] = args.repeats
    if args.folds is not None:
        over["n_folds"] = args.folds
    if args.epochs is not None:
        over["epochs"] = args.epochs
    if over:
        cfg = replace(cfg, **over)

    print(f"task={cfg.task} norm={cfg.norm_mode} threads={cfg.threads} "
          f"hash={cfg.hash()}")
    print(f"outdir={args.outdir or cfg.outdir()}")
    run(cfg, outdir=args.outdir)


if __name__ == "__main__":
    main()
