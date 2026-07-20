import os
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd_list):
    print(f"\n> {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"ERROR executing command: {' '.join(cmd_list)}")
        print(result.stderr)
        exit(1)
    print(result.stdout)
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Methodical testing script for EOT models.")
    parser.add_argument("--train_dirs", nargs="+", default=["eot_data/english", "eot_data/hindi"], help="Folders to train on")
    parser.add_argument("--test_dirs", nargs="+", default=["eot_data/english", "eot_data/hindi"], help="Folders to test on")
    parser.add_argument("--out_dir", default="outputs", help="Directory to store predictions")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if model exists")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.out_dir).mkdir(exist_ok=True)

    print("=" * 60)
    model_path = Path(args.out_dir).parent / "eot_model3.pkl"
    if model_path.exists() and not args.force_train:
        print(f"STEP 1: SKIPPED. Model '{model_path}' already exists.")
        print("Pass --force_train to re-train the model.")
    else:
        print(f"STEP 1: Training Model on {args.train_dirs}")
        print("=" * 60)
        train_cmd = ["python", "baseline-approach3.py", "--data_dirs"] + args.train_dirs + ["--out_model_dir", "."]
        run_cmd(train_cmd)

    print("=" * 60)
    print("STEP 2: Evaluating Model")
    print("=" * 60)
    
    for test_dir in args.test_dirs:
        lang_name = Path(test_dir).name
        pred_file = os.path.join(args.out_dir, f"predictions_{lang_name}.csv")
        
        print(f"\n--- Testing on: {lang_name.upper()} ---")
        
        # 1. Predict
        predict_cmd = ["python", "predict.py", "--data_dir", test_dir, "--model_dir", ".", "--out", pred_file]
        run_cmd(predict_cmd)
        
        # 2. Score
        score_cmd = ["python", "score.py", "--data_dir", test_dir, "--pred", pred_file]
        run_cmd(score_cmd)
        
    print("=" * 60)
    print(f"All done! Predictions saved in the '{args.out_dir}/' directory.")

if __name__ == "__main__":
    main()
