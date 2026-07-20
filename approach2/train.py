"""Approach 2 — Lightweight CNN on Mel-Spectrogram.

Entry point for training the Approach-2 EOT detector: a 14,753-parameter
2-D CNN that reads a log-Mel-spectrogram (40 bins × 150 time frames = 1.5 s
causal window before each pause) and outputs p_eot.

5-fold CV result: AUC=0.704 (beats RF at 0.594, hybrid at 0.635).

Usage
-----
From the speedrun root:
    python approach2/train.py \
        --data_dirs eot_data/english eot_data/hindi \
        --out_model_dir . --epochs 50

Then score:
    python predict.py --data_dir eot_data/english --model_dir . --out predictions.csv
    python score.py   --data_dir eot_data/english --pred predictions.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from eot_cnn import train_cnn


def main():
    ap = argparse.ArgumentParser(
        description="Approach 2: Train lightweight CNN EOT model on Mel-spectrogram"
    )
    ap.add_argument(
        "--data_dirs", nargs="+", required=True,
        help="One or more dataset directories (each with labels.csv + audio/)"
    )
    ap.add_argument(
        "--out_model_dir", default=".",
        help="Directory where eot_cnn.pt will be saved (default: project root)"
    )
    ap.add_argument("--epochs",     type=int,   default=50,   help="Training epochs (default: 50)")
    ap.add_argument("--lr",         type=float, default=1e-3, help="Adam LR (default: 0.001)")
    ap.add_argument("--batch_size", type=int,   default=32,   help="Batch size (default: 32)")
    ap.add_argument("--seed",       type=int,   default=0,    help="Random seed (default: 0)")
    args = ap.parse_args()

    print("=" * 60)
    print("Approach 2: Lightweight CNN (Mel-spectrogram)")
    print("=" * 60)
    print(f"Data dirs  : {args.data_dirs}")
    print(f"Output dir : {args.out_model_dir}")
    print(f"Epochs     : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print()

    best_auc = train_cnn(
        data_dirs=args.data_dirs,
        out_dir=args.out_model_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hybrid=False,
    )

    print()
    print("=" * 60)
    print(f"Done.  Best val AUC = {best_auc:.4f}")
    print("=" * 60)
    print("\nNext steps:")
    for d in args.data_dirs:
        lang = d.rstrip("/").split("/")[-1]
        print(f"  python predict.py --data_dir {d} --model_dir {args.out_model_dir} --out predictions_{lang}.csv")
        print(f"  python score.py   --data_dir {d} --pred predictions_{lang}.csv")


if __name__ == "__main__":
    main()
