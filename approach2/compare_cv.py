"""Cross-validation comparison: RF vs CNN vs Geometric Hybrid.

Run from the speedrun root:
    python approach2/compare_cv.py

Runs group-aware 5-fold CV (split by turn_id) and prints mean AUC for
each approach. Used to decide whether to blend RF + CNN probabilities.

Result on EN+HI combined:
    CNN    AUC: 0.704  (winner)
    Hybrid AUC: 0.635
    RF     AUC: 0.594
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from eot_model import extract_features, load_audio
from eot_cnn import mel_features, EOTConvNet

import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_DIRS = ["eot_data/english", "eot_data/hindi"]


def load_all_rows(data_dirs):
    rows = []
    audio_cache = {}
    for d in data_dirs:
        with open(os.path.join(d, "labels.csv")) as f:
            for r in csv.DictReader(f):
                ap = os.path.join(d, r["audio_file"])
                if ap not in audio_cache:
                    audio_cache[ap] = load_audio(ap)
                x, sr = audio_cache[ap]
                rows.append({
                    "turn_id":    r["turn_id"],
                    "rf_feat":    extract_features(x, sr, float(r["pause_start"])),
                    "mel_feat":   mel_features(x, sr, float(r["pause_start"])),
                    "label":      1 if r["label"] == "eot" else 0,
                })
    return rows


def train_fold_cnn(tr_specs, tr_labels, va_specs, va_labels, epochs=25):
    """Train a fresh CNN on one fold and return val probabilities."""
    n_pos = sum(tr_labels); n_neg = len(tr_labels) - n_pos
    pw = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    model = EOTConvNet(return_embedding=False)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_ds = TensorDataset(
        torch.stack([torch.from_numpy(s).unsqueeze(0) for s in tr_specs]),
        torch.tensor(tr_labels, dtype=torch.float32),
    )
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)

    best_auc, best_state = 0.0, None
    va_t = torch.stack([torch.from_numpy(s).unsqueeze(0) for s in va_specs])

    for _ in range(epochs):
        model.train()
        for specs, lbs in tr_dl:
            opt.zero_grad(); crit(model(specs), lbs).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(va_t)).numpy()
        auc = roc_auc_score(va_labels, probs) if len(set(va_labels)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(va_t)).numpy()


def main():
    print("Loading features for all pauses...")
    rows = load_all_rows(DATA_DIRS)

    turn_ids = [r["turn_id"] for r in rows]
    turns    = list(dict.fromkeys(turn_ids))   # deduplicated, ordered
    random.seed(42); random.shuffle(turns)
    folds = [turns[i::5] for i in range(5)]

    rf_aucs, cnn_aucs, hyb_aucs = [], [], []

    for fold_i, val_turns in enumerate(folds):
        val_set = set(val_turns)
        tr = [r for r in rows if r["turn_id"] not in val_set]
        va = [r for r in rows if r["turn_id"] in val_set]
        if not va:
            continue

        tr_rf = np.array([r["rf_feat"] for r in tr])
        va_rf = np.array([r["rf_feat"] for r in va])
        tr_lb = np.array([r["label"]   for r in tr])
        va_lb = np.array([r["label"]   for r in va])

        # RF
        pipe = Pipeline([("sc", StandardScaler()),
                         ("rf", RandomForestClassifier(
                             n_estimators=200, max_depth=8,
                             class_weight="balanced", random_state=0))])
        pipe.fit(tr_rf, tr_lb)
        rf_prob = pipe.predict_proba(va_rf)[:, 1]

        # CNN
        cnn_prob = train_fold_cnn(
            [r["mel_feat"] for r in tr], list(tr_lb),
            [r["mel_feat"] for r in va], list(va_lb),
        )

        # Geometric blend
        hyb_prob = np.sqrt(rf_prob * cnn_prob)

        ra  = roc_auc_score(va_lb, rf_prob)  if len(set(va_lb)) > 1 else 0.5
        ca  = roc_auc_score(va_lb, cnn_prob) if len(set(va_lb)) > 1 else 0.5
        ha  = roc_auc_score(va_lb, hyb_prob) if len(set(va_lb)) > 1 else 0.5
        rf_aucs.append(ra); cnn_aucs.append(ca); hyb_aucs.append(ha)
        print(f"Fold {fold_i+1}: RF={ra:.3f}  CNN={ca:.3f}  Hybrid={ha:.3f}")

    print()
    print(f"Mean RF     AUC: {np.mean(rf_aucs):.3f}")
    print(f"Mean CNN    AUC: {np.mean(cnn_aucs):.3f}")
    print(f"Mean Hybrid AUC: {np.mean(hyb_aucs):.3f}")
    winner = max(zip([np.mean(rf_aucs), np.mean(cnn_aucs), np.mean(hyb_aucs)],
                     ["RF", "CNN", "Hybrid"]), key=lambda x: x[0])
    print(f"\nWinner: {winner[1]} (AUC={winner[0]:.3f})")


if __name__ == "__main__":
    main()
