# Approach 2 — Lightweight CNN on Mel-Spectrogram

## What It Does

A **14,753-parameter 2-D CNN** reads the last 1.5 s of speech before each pause as a **log-Mel-spectrogram** (40 mel bins × 150 time frames at 10 ms hop) and outputs p_eot. Learns spectral texture patterns — formant trailing, creak, energy fade-out — that handcrafted prosodic features miss.

### Architecture
```
Input: (1, 40, 150)   ← 1 channel, 40 mel bins, 150 time frames

Conv2d(1→16, 3×3) + BN + ReLU + MaxPool(2×2)    → (16, 20, 75)
Conv2d(16→32, 3×3) + BN + ReLU + MaxPool(2×2)   → (32, 10, 37)
Conv2d(32→32, 3×3) + BN + ReLU + AdaptiveAvgPool → (32, 1, 1)
Linear(32→16) + ReLU + Dropout(0.4)
Linear(16→1) → sigmoid → p_eot
```
**Total: 14,753 parameters** — CPU inference < 1 ms per pause.

### Training Details
- `BCEWithLogitsLoss` with `pos_weight = n_hold / n_eot` for class imbalance
- Adam lr=0.001, weight_decay=1e-4, cosine LR annealing
- Augmentation: ±3 dB amplitude jitter + ±50 ms time shift (in log-mel domain)
- 80/20 turn-level train/val split (no leakage between turns)
- Trained on English + Hindi combined (496 pauses, 200 turns)

### Files
| File | Purpose |
|---|---|
| `train.py` | Training entry point |
| `compare_cv.py` | 5-fold CV: RF vs CNN vs Hybrid comparison |
| `../../eot_cnn.py` | CNN model definition + `train_cnn()` + `load_cnn()` + `predict_cnn()` |

## Key Findings

| Model | 5-fold CV AUC | Winner |
|---|---|---|
| **CNN (Mel-spectrogram)** | **0.704** | ✅ |
| Geometric Hybrid (RF × CNN) | 0.635 | |
| RF (16 prosodic features) | 0.594 | |

CNN outperforms RF by +11 AUC points. Geometric blending hurts because the RF's probability noise floor dilutes the CNN's sharper signal.

## Usage

From the `speedrun/` root:
```bash
# Train
python approach2/train.py \
    --data_dirs eot_data/english eot_data/hindi \
    --out_model_dir . --epochs 50

# Predict + Score (English)
python predict.py --data_dir eot_data/english --model_dir . --out predictions.csv
python score.py   --data_dir eot_data/english --pred predictions.csv

# Predict + Score (Hindi)
python predict.py --data_dir eot_data/hindi --model_dir . --out predictions_hindi.csv
python score.py   --data_dir eot_data/hindi --pred predictions_hindi.csv

# Run CV comparison
python approach2/compare_cv.py
```

## Scored Results
| Language | AUC | Delay | Interrupted |
|---|---|---|---|
| English | 0.598 | 1429 ms | 5.0% |
| Hindi   | 0.643 | 850 ms  | 5.0% |
| Baseline | — | 1600 ms | 0.0% |
