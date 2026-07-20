import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

from features import f0_contour, frame_energy_db, load_wav, speech_before

MODEL_FILENAME = "eot_model3.pkl"

class HeuristicEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classes_ = None

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)
        for i in range(X.shape[0]):
            duration = X[i, 0]
            if duration < 0.200:
                # Force prediction to 'Hold' (p_eot = 0.0)
                probs[i, 0] = 1.0
                probs[i, 1] = 0.0
        return probs

def extract_features(
    x: np.ndarray, sr: int, pause_start: float, window_s: float = 1.5
) -> np.ndarray:
    seg = speech_before(x, sr, pause_start, window_s=window_s)
    if len(seg) < sr // 10:
        return np.zeros(16, dtype=np.float32)

    energy = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    voiced_mask = (f0 > 0).astype(np.float32)

    def slope(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        x_idx = np.arange(values.size, dtype=np.float32)
        return float(np.polyfit(x_idx, values.astype(np.float32), 1)[0])

    def last_run(mask: np.ndarray) -> int:
        count = 0
        for item in mask[::-1]:
            if item:
                count += 1
            else:
                break
        return count

    energy_tail = energy[-10:]
    f0_tail = f0[-10:]

    features = np.array(
        [
            len(seg) / sr,
            len(energy),
            float(np.mean(energy_tail)) if energy_tail.size else 0.0,
            float(np.std(energy_tail)) if energy_tail.size else 0.0,
            slope(energy_tail),
            float(np.mean(voiced)) if voiced.size else 0.0,
            float(np.std(voiced)) if voiced.size else 0.0,
            slope(voiced[-10:]) if voiced.size >= 2 else 0.0,
            float(np.mean(f0_tail)) if f0_tail.size else 0.0,
            float(np.std(f0_tail)) if f0_tail.size else 0.0,
            slope(f0_tail),
            float(np.mean(voiced_mask)),
            float(last_run(voiced_mask)) / max(1.0, float(len(voiced_mask))),
            float(np.percentile(energy, 10)) if energy.size else 0.0,
            float(np.percentile(energy, 90)) if energy.size else 0.0,
            float(np.max(energy) - np.min(energy)) if energy.size else 0.0,
        ],
        dtype=np.float32,
    )
    return features

def load_audio(path: str):
    return load_wav(path)

def make_classifier() -> HeuristicEnsemble:
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=5, class_weight="balanced", random_state=42
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb)],
        voting='soft'
    )
    
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ]
    )
    
    return HeuristicEnsemble(pipeline)

@dataclass
class EOTArtifact:
    model: HeuristicEnsemble

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

def save_artifact(model: HeuristicEnsemble, model_dir: str | os.PathLike[str]) -> Path:
    model_path = Path(model_dir) / MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(EOTArtifact(model=model), f)
    return model_path

def load_artifact(model_dir: str | os.PathLike[str] | None):
    if model_dir is None:
        return None
    model_path = Path(model_dir) / MODEL_FILENAME
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    if isinstance(artifact, EOTArtifact):
        return artifact
    return None
