"""Baseline — Silence-Only EOT detector (starter skeleton).

The original assignment starter: logs every pause as p_eot=1.0 (always fire).
Scores 1600 ms mean response delay at 0% interrupted turns on English.

This is the reference point. Every approach above this is an improvement.

Usage
-----
From the speedrun root:
    python baseline/run.py --data_dir eot_data/english --out predictions.csv
    python score.py        --data_dir eot_data/english --pred predictions.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Baseline: always predict p_eot=1.0")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.data_dir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "turn_id":     r["turn_id"],
                "pause_index": r["pause_index"],
                "p_eot":       1.0,
            })

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions -> {out_path}  (baseline: all p_eot=1.0)")


if __name__ == "__main__":
    main()
