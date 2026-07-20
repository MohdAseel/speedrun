import argparse
import csv
import os
from pathlib import Path
from features import load_wav

def predict_rows(data_dir: str, model_type: str, model_obj, ext_fn=None):
    labels_path = os.path.join(data_dir, "labels.csv")
    cache = {}

    with open(labels_path) as f:
        for row in csv.DictReader(f):
            turn_id    = row["turn_id"]
            pause_idx  = row["pause_index"]
            pause_start = float(row["pause_start"])
            audio_path  = os.path.join(data_dir, row["audio_file"])

            if audio_path not in cache:
                cache[audio_path] = load_wav(audio_path)
            x, sr = cache[audio_path]

            if model_type == "cnn":
                from eot_cnn import predict_cnn
                p_eot = predict_cnn(model_obj, x, sr, pause_start)
            elif model_type in ["rf", "heuristic"] and ext_fn is not None:
                feats = ext_fn(x, sr, pause_start).reshape(1, -1)
                p_eot = float(model_obj.predict_proba(feats)[0, 1])
            else:
                p_eot = 1.0

            yield {
                "turn_id":     turn_id,
                "pause_index": pause_idx,
                "p_eot":       p_eot,
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out",       default="predictions.csv")
    parser.add_argument("--model_dir", default=".")
    parser.add_argument("--force_model", default=None, choices=["cnn", "heuristic", "rf"], help="Force a specific model to load")
    args = parser.parse_args()

    model_type = None
    model_obj = None
    ext_fn = None

    model_path_cnn = os.path.join(args.model_dir, "eot_cnn.pt")
    model_path3 = os.path.join(args.model_dir, "eot_model3.pkl")
    model_path1 = os.path.join(args.model_dir, "eot_model.pkl")

    if (args.force_model == "heuristic" or not args.force_model) and os.path.exists(model_path3):
        from eot_model3 import load_artifact, extract_features
        model_obj = load_artifact(args.model_dir)
        ext_fn = extract_features
        model_type = "heuristic"
        print(f"Loaded Approach 3 (Heuristic Ensemble) from {args.model_dir}")
    elif (args.force_model == "cnn" or not args.force_model) and os.path.exists(model_path_cnn):
        from eot_cnn import load_cnn
        model_obj = load_cnn(args.model_dir)
        model_type = "cnn"
        print(f"Loaded Approach 2 (CNN) from {args.model_dir}")
    elif (args.force_model == "rf" or not args.force_model) and os.path.exists(model_path1):
        from eot_model import load_artifact, extract_features
        model_obj = load_artifact(args.model_dir)
        ext_fn = extract_features
        model_type = "rf"
        print(f"Loaded Approach 1 (Random Forest) from {args.model_dir}")
    else:
        print("No trained model found, falling back to p_eot=1.0 (silence baseline).")

    rows = list(predict_rows(args.data_dir, model_type=model_type, model_obj=model_obj, ext_fn=ext_fn))

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[predict] Wrote {len(rows)} predictions -> {out_path}")

if __name__ == "__main__":
    main()
