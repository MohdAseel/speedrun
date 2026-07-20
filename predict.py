"""Submission inference entry point for the EOT task.

This is the stable CLI the grader expects:

    python predict.py --data_dir <folder> --out predictions.csv

For now this shell keeps the interface fixed and uses the same causal
audio-loading path as the starter code. The model-loading hook is in place
so we can drop in a trained classifier next without changing the grader
contract.
"""

import argparse
import csv
import os
from pathlib import Path

from eot_model import load_artifact


def predict_rows(data_dir: str):
    labels_path = os.path.join(data_dir, "labels.csv")
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            yield {
                "turn_id": row["turn_id"],
                "pause_index": row["pause_index"],
                "p_eot": 1.0,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()

    _model = load_artifact(args.model_dir)
    rows = list(predict_rows(args.data_dir))

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
