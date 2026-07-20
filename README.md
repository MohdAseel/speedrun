# EOT (End-of-Turn) Detection Model

Detects whether a pause in a voice conversation marks the end of the user's turn (`eot`) or a mid-speech hold (`hold`). Built from scratch — no pretrained models.

---

## Project Structure

```
speedrun/
├── approach1/               ← Approach 1: Classical ML (Random Forest)
│   ├── train.py             ← Training script
│   └── README.md            ← Feature list, results, usage
│
├── approach2/               ← Approach 2: Lightweight CNN (BEST)
│   ├── train.py             ← Training script
│   ├── compare_cv.py        ← 5-fold CV: RF vs CNN vs Hybrid
│   └── README.md            ← Architecture, results, usage
│
├── baseline/                ← Reference baseline (silence-only, 1600 ms)
│   ├── run.py               ← Generates p_eot=1.0 predictions
│   └── README.md
│
├── eot_data/
│   ├── english/             ← 100 turns, labels.csv, audio/en__*.wav
│   └── hindi/               ← 100 turns, labels.csv, audio/hi__*.wav
│
├── outputs/                 ← Archived prediction CSVs
│
│── Shared modules (used by all approaches + grader)
├── features.py              ← Audio loading, speech_before, frame_energy_db, f0_contour
├── eot_model.py             ← Approach 1 model: extract_features, RandomForest pipeline
├── eot_cnn.py               ← Approach 2 model: EOTConvNet, train_cnn, load_cnn, predict_cnn
│
│── Grader-facing (stable CLI contract)
├── predict.py               ← python predict.py --data_dir <dir> --out predictions.csv
├── score.py                 ← python score.py   --data_dir <dir> --pred predictions.csv
│
│── Saved model artifacts
├── eot_cnn.pt               ← Approach 2 CNN weights (active model)
├── eot_model.pkl            ← Approach 1 RF weights (fallback)
│
│── Deliverables
├── predictions.csv          ← English predictions (Approach 2 CNN)
├── predictions_hindi.csv    ← Hindi predictions (Approach 2 CNN)
├── SUMMARY.html             ← Full solution writeup
├── RUNLOG.md                ← Score log for every run
└── NOTES.md                 ← Model signal, failures, future work
```

---

## Quick Start

### 1. Train (Approach 2 — CNN, recommended)
```bash
python approach2/train.py \
    --data_dirs eot_data/english eot_data/hindi \
    --out_model_dir . --epochs 50
```

### 2. Predict
```bash
python predict.py --data_dir eot_data/english --model_dir . --out predictions.csv
python predict.py --data_dir eot_data/hindi   --model_dir . --out predictions_hindi.csv
```

### 3. Score
```bash
python score.py --data_dir eot_data/english --pred predictions.csv
python score.py --data_dir eot_data/hindi   --pred predictions_hindi.csv
```

---

## Results Summary

| Approach | Model | CV AUC | English Delay | Hindi Delay |
|---|---|---|---|---|
| Baseline | Silence (p_eot=1.0) | 0.514 | 1600 ms | 850 ms |
| Approach 1 | Random Forest (16 prosodic features) | 0.594 | — | — |
| **Approach 2** | **CNN (Mel-spectrogram, 14k params)** | **0.704** | **1429 ms** | **850 ms** |

### Model Loading Priority (`predict.py`)
1. `eot_cnn.pt` — Approach 2 CNN (best)
2. `eot_model.pkl` — Approach 1 RF (fallback)
3. `p_eot = 1.0` — Silence baseline (last resort)

---

## Causality Contract

Every feature only touches `audio[0 : pause_start]`. `pause_end` is **never** used in feature computation — compliant for live voice agents.
