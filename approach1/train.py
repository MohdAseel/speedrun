"""Approach 1 — Classical ML with Handcrafted Prosodic Features.

Trains a Random Forest on 16 causal prosodic features extracted from the
last 1.5 s of speech before each pause: energy tail stats, F0 slope,
voiced ratio, last-run length, energy range, percentiles.

Usage
-----
From the speedrun root:
    python approach1/train.py \
        --data_dirs eot_data/english eot_data/hindi \
        --out_model_dir .

Then score:
    python predict.py --data_dir eot_data/english --model_dir . --out predictions.csv
    python score.py   --data_dir eot_data/english --pred predictions.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import numpy as np

from eot_model import load_audio, extract_features, make_classifier, save_artifact


def main():
    ap = argparse.ArgumentParser(
        description="Approach 1: Train Random Forest on prosodic features"
    )
    ap.add_argument(
        "--data_dirs", nargs="+", required=True,
        help="List of dataset directories (each with labels.csv + audio/)"
    )
    ap.add_argument("--out_model_dir", default=".", help="Where to save eot_model.pkl")
    args = ap.parse_args()

    rows = []
    for d in args.data_dirs:
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            r["_data_dir"] = d
            rows.append(r)

    cache = {}
    X, y = [], []
    print(f"Extracting features for {len(rows)} pauses across {len(args.data_dirs)} datasets...")
    for i, r in enumerate(rows):
        path = os.path.join(r["_data_dir"], r["audio_file"])
        if path not in cache:
            cache[path] = load_audio(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(rows)}")

    X, y = np.array(X), np.array(y)
    print(f"Training RandomForest on {len(X)} samples...")
    clf = make_classifier()
    clf.fit(X, y)

    save_path = save_artifact(clf, args.out_model_dir)
    print(f"Saved -> {save_path}")
    print("\nNext:")
    print(f"  python predict.py --data_dir {args.data_dirs[0]} --model_dir {args.out_model_dir}")


if __name__ == "__main__":
    main()
