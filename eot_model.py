"""Shared model utilities for end-of-turn detection.

This module centralizes the causal feature extraction path so training and
inference can stay in sync. Every feature only touches audio strictly before
pause_start.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import f0_contour, frame_energy_db, load_wav, speech_before

MODEL_FILENAME = "eot_model.pkl"


def _safe_stats(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
        ],
        dtype=np.float32,
    )


def extract_features(
    x: np.ndarray, sr: int, pause_start: float, window_s: float = 1.5
) -> np.ndarray:
    """Return causal features from audio up to pause_start."""
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


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


@dataclass
class EOTArtifact:
    model: Pipeline

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)


def save_artifact(model: Pipeline, model_dir: str | os.PathLike[str]) -> Path:
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
