#!/usr/bin/env python3
"""
Push CV accuracy higher without labeling new data.

Runs the same 5-fold stratified CV as train_bark_cnn.py, then layers three
cheap accuracy improvements on top of the baseline:

  1. Persist every fold's model (+ its normalization stats) so inference
     code can average their probabilities — typically 0.5-2 pp.
  2. Test-time augmentation (TTA): average sigmoids over small time shifts
     of the input spectrogram — typically 0.5-1 pp.
  3. Threshold tuning: sweep the decision threshold on the OOF probs
     (post-TTA) to find the accuracy-optimal point. 0.5 is rarely
     optimal for imbalanced sets.

Also writes a list of the clips that are still misclassified after all
three tweaks, sorted by confidence — these are the ones worth listening
to when deciding where to invest new labels.

Artifacts:
  fold_models/fold_{0..4}.pt   — each has state_dict, mean, std
  fold_models/fold_meta.json   — ensemble config for inference
  bark_cnn_threshold.json      — tuned decision threshold
  misclassified.csv            — review list
  tuning_report.txt            — before/after accuracy + confusion matrices

Runtime: roughly the same as train_bark_cnn.py (it retrains all 5 folds).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from train_bark_cnn import (
    AudioCfg,
    BarkCNN,
    load_clip,
    logmel,
    normalize_specs,
    predict_probs,
    read_labels,
    train_one,
)
from label_barks import parse_log_dbfs

PROJECT = Path(__file__).resolve().parent
FOLD_DIR = PROJECT / "fold_models"
MISCLASS_CSV = PROJECT / "misclassified.csv"
REPORT_PATH = PROJECT / "tuning_report.txt"
THRESHOLD_PATH = PROJECT / "bark_cnn_threshold.json"

TTA_SHIFTS = (-8, -4, 0, 4, 8)  # frame shifts along the mel time axis


def tta_predict(model, X: torch.Tensor, shifts=TTA_SHIFTS) -> np.ndarray:
    """Average sigmoid predictions over time-shifted copies of X."""
    model.eval()
    acc = None
    with torch.no_grad():
        for s in shifts:
            xs = X if s == 0 else torch.roll(X, shifts=int(s), dims=-1)
            p = torch.sigmoid(model(xs)).cpu().numpy()
            acc = p if acc is None else acc + p
    return acc / len(shifts)


def sweep_threshold(y: np.ndarray, probs: np.ndarray, metric: str) -> tuple[float, float]:
    ts = np.linspace(0.05, 0.95, 181)
    best_t, best_v = 0.5, -1.0
    for t in ts:
        preds = (probs >= t).astype(int)
        v = float((preds == y).mean()) if metric == "accuracy" else float(f1_score(y, preds))
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t, best_v


def main():
    cfg = AudioCfg()
    print(f"[config] {cfg}")

    paths, y = read_labels()
    print(f"[data] {len(paths)} clips  ({int((y == 1).sum())} YES / {int((y == 0).sum())} NO)")

    print("[feat] computing log-mel spectrograms...")
    specs = np.stack([logmel(load_clip(p, cfg), cfg) for p in paths])
    print(f"[feat] specs shape {specs.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    FOLD_DIR.mkdir(exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_plain = np.zeros(len(y), dtype=np.float32)
    oof_tta = np.zeros(len(y), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(specs, y)):
        X_tr_np, m, s = normalize_specs(specs[tr])
        X_va_np, _, _ = normalize_specs(specs[va], m, s)
        X_tr = torch.from_numpy(X_tr_np).unsqueeze(1).to(device)
        X_va = torch.from_numpy(X_va_np).unsqueeze(1).to(device)
        y_tr = torch.from_numpy(y[tr].astype(np.float32)).to(device)
        y_va = torch.from_numpy(y[va].astype(np.float32)).to(device)

        n_pos = int((y[tr] == 1).sum())
        n_neg = int((y[tr] == 0).sum())
        pw = n_neg / max(n_pos, 1)

        model = BarkCNN().to(device)
        train_one(model, X_tr, y_tr, X_va, y_va, device, epochs=60, pos_weight=pw)

        probs_plain = predict_probs(model, X_va)
        probs_tta = tta_predict(model, X_va)
        oof_plain[va] = probs_plain
        oof_tta[va] = probs_tta

        torch.save(
            {"state_dict": model.state_dict(), "mean": float(m), "std": float(s)},
            FOLD_DIR / f"fold_{fold}.pt",
        )
        acc_p = float(((probs_plain >= 0.5) == y[va]).mean())
        acc_t = float(((probs_tta >= 0.5) == y[va]).mean())
        print(f"[fold {fold+1}/5] acc plain={acc_p:.3f}  +TTA={acc_t:.3f}  "
              f"(n_val={len(va)})", flush=True)

    base_acc = float(((oof_plain >= 0.5) == y).mean())
    tta_acc = float(((oof_tta >= 0.5) == y).mean())
    best_t_acc, best_v_acc = sweep_threshold(y, oof_tta, "accuracy")
    best_t_f1, best_v_f1 = sweep_threshold(y, oof_tta, "f1")

    cm_base = confusion_matrix(y, (oof_plain >= 0.5).astype(int), labels=[0, 1])
    cm_tta = confusion_matrix(y, (oof_tta >= 0.5).astype(int), labels=[0, 1])
    cm_tuned = confusion_matrix(y, (oof_tta >= best_t_acc).astype(int), labels=[0, 1])

    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": round(best_t_acc, 4),
        "source": "tune_accuracy.py",
        "oof_acc_at_threshold": round(best_v_acc, 4),
    }, indent=2))
    print(f"[out] wrote {THRESHOLD_PATH}  threshold={best_t_acc:.3f}")

    (FOLD_DIR / "fold_meta.json").write_text(json.dumps({
        "n_folds": 5,
        "audio": asdict(cfg),
        "use_tta": True,
        "tta_shifts": list(TTA_SHIFTS),
        "note": "each fold_*.pt has its own mean/std; normalize before forward pass, then average sigmoids across folds",
    }, indent=2))

    log_dbfs = parse_log_dbfs()
    preds_tuned = (oof_tta >= best_t_acc).astype(int)
    miss_rows = []
    for p, yt, yp, pr in zip(paths, y, preds_tuned, oof_tta):
        if yt == yp:
            continue
        miss_rows.append({
            "filename": p.name,
            "true_label": "YES" if yt == 1 else "NO",
            "pred_label": "YES" if yp == 1 else "NO",
            "oof_prob": round(float(pr), 4),
            "margin": round(abs(float(pr) - best_t_acc), 4),
            "peak_dbfs": log_dbfs.get(p.name, ""),
            "error_type": "FP" if yp == 1 else "FN",
        })
    miss_rows.sort(key=lambda r: r["margin"])
    with MISCLASS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(miss_rows[0].keys()) if miss_rows
                           else ["filename", "true_label", "pred_label",
                                 "oof_prob", "margin", "peak_dbfs", "error_type"])
        w.writeheader()
        w.writerows(miss_rows)
    print(f"[out] wrote {MISCLASS_CSV}  ({len(miss_rows)} errors)")

    lines = [
        "=== OOF accuracy ===",
        f"  baseline (plain, t=0.5)           : {base_acc:.4f}",
        f"  + TTA (t=0.5)                     : {tta_acc:.4f}",
        f"  + TTA + tuned threshold (t={best_t_acc:.3f}) : {best_v_acc:.4f}",
        f"  (F1-optimal threshold {best_t_f1:.3f} => F1 {best_v_f1:.4f})",
        "",
        "=== Confusion matrices (rows=NO/YES true, cols=NO/YES pred) ===",
        f"  baseline: {cm_base.tolist()}",
        f"  +TTA    : {cm_tta.tolist()}",
        f"  tuned   : {cm_tuned.tolist()}",
        "",
        f"Saved fold ensemble to {FOLD_DIR}/ ({sum(1 for _ in FOLD_DIR.glob('fold_*.pt'))} models).",
        "Inference code can average sigmoids over all 5 folds for a further bump.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[out] wrote {REPORT_PATH}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
