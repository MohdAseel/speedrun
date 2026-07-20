"""Compare two EOT model candidates on the same labeled dataset.

This utility supports two workflows:

1. Compare saved model artifacts by passing --model_dir_a and --model_dir_b.
2. Compare existing prediction CSVs by passing --pred_a and --pred_b.

For model artifacts, the script rebuilds predictions causally from the audio
and then runs the official scorer on the generated CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from eot_model import extract_features, load_artifact, load_audio
from score import score as score_predictions


def load_labels(data_dir: str):
    labels_path = os.path.join(data_dir, "labels.csv")
    with open(labels_path) as f:
        yield from csv.DictReader(f)


def write_predictions_from_model(data_dir: str, model_dir: str, out_path: Path):
    artifact = load_artifact(model_dir)
    if artifact is None:
        raise SystemExit(f"no saved model artifact found in {model_dir}")

    cache = {}
    rows = []
    for row in load_labels(data_dir):
        audio_path = os.path.join(data_dir, row["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_audio(audio_path)
        x, sr = cache[audio_path]
        features = extract_features(x, sr, float(row["pause_start"]))
        probability = float(artifact.predict_proba(features.reshape(1, -1))[0, 1])
        rows.append(
            {
                "turn_id": row["turn_id"],
                "pause_index": row["pause_index"],
                "p_eot": f"{probability:.6f}",
            }
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(score_a, score_b):
    print("Model comparison")
    print(f"  A: delay={score_a['latency']*1000:.0f} ms, interrupted={score_a['cutoff']*100:.1f}%, auc={score_a['auc']:.3f}")
    print(f"  B: delay={score_b['latency']*1000:.0f} ms, interrupted={score_b['cutoff']*100:.1f}%, auc={score_b['auc']:.3f}")
    winner = "A" if score_a["latency"] < score_b["latency"] else "B" if score_b["latency"] < score_a["latency"] else "tie"
    print(f"  winner: {winner}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir_a", default=None)
    parser.add_argument("--model_dir_b", default=None)
    parser.add_argument("--pred_a", default=None)
    parser.add_argument("--pred_b", default=None)
    args = parser.parse_args()

    using_models = args.model_dir_a or args.model_dir_b
    using_predictions = args.pred_a or args.pred_b
    if using_models and using_predictions:
        raise SystemExit("use either model dirs or prediction files, not both")
    if not using_models and not using_predictions:
        raise SystemExit("provide either --model_dir_* or --pred_*")

    labels_path = os.path.join(args.data_dir, "labels.csv")

    if using_models:
        if not args.model_dir_a or not args.model_dir_b:
            raise SystemExit("both --model_dir_a and --model_dir_b are required")
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_a = Path(tmpdir) / "pred_a.csv"
            pred_b = Path(tmpdir) / "pred_b.csv"
            write_predictions_from_model(args.data_dir, args.model_dir_a, pred_a)
            write_predictions_from_model(args.data_dir, args.model_dir_b, pred_b)
            score_a = score_predictions(labels_path, pred_a)
            score_b = score_predictions(labels_path, pred_b)
            summarize(score_a, score_b)
        return

    if not args.pred_a or not args.pred_b:
        raise SystemExit("both --pred_a and --pred_b are required")
    score_a = score_predictions(labels_path, args.pred_a)
    score_b = score_predictions(labels_path, args.pred_b)
    summarize(score_a, score_b)


if __name__ == "__main__":
    main()