import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import random
import numpy as np

from eot_model3 import load_audio, extract_features, save_artifact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True, help="List of dataset directories")
    ap.add_argument("--out_model_dir", default=".")
    args = ap.parse_args()

    rows = []
    for d in args.data_dirs:
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            r["_data_dir"] = d
            rows.append(r)
            
    random.seed(42)
    random.shuffle(rows)
    
    cache = {}
    X, y = [], []
    
    print(f"Extracting features for {len(rows)} pauses (Approach 3: Heuristic Ensemble)...")
    for i, r in enumerate(rows):
        path = os.path.join(r["_data_dir"], r["audio_file"])
        if path not in cache:
            cache[path] = load_audio(path)
        x, sr = cache[path]
        
        feats = extract_features(x, sr, float(r["pause_start"]))
        X.append(feats)
        y.append(1 if r["label"] == "eot" else 0)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(rows)}...")

    X = np.array(X)
    y = np.array(y)

    from sklearn.model_selection import GridSearchCV
    from eot_model3 import make_classifier_pipeline, HeuristicEnsemble
    
    print("Running GridSearchCV for Approach 3 Ensemble...")
    pipeline = make_classifier_pipeline()
    
    # We test combinations of RF depth/estimators and HGB iterations/learning rate
    param_grid = {
        'clf__rf__n_estimators': [50, 100],
        'clf__rf__max_depth': [None, 6, 8],
        'clf__hgb__max_iter': [50, 100],
        'clf__hgb__learning_rate': [0.05, 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X, y)
    
    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Best Cross-Validation AUC: {grid.best_score_:.4f}")
    
    # Wrap the highly-optimized best estimator in our rule-based HeuristicEnsemble
    final_model = HeuristicEnsemble(grid.best_estimator_)
    final_model.fit(X, y)
    
    save_path = save_artifact(final_model, args.out_model_dir)
    print(f"Model successfully trained and saved to: {save_path}")

if __name__ == "__main__":
    main()
