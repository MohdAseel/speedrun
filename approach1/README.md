# Approach 1 — Classical ML with Handcrafted Prosodic Features

## What It Does

Extracts **16 causal prosodic features** from the last 1.5 s of speech before each pause and trains a `RandomForestClassifier` (100 trees, max_depth=6, class-weight balanced) with `StandardScaler` preprocessing.

### Features (from `eot_model.extract_features`)
| Feature | Rationale |
|---|---|
| Segment duration | Short bursts = backchannels (likely EOT) |
| Frame count | Speaking rate proxy |
| Energy tail mean / std | Smooth fade = EOT; abrupt cut = hold |
| Energy tail slope | Decay into pause |
| Voiced F0 mean / std | Pitch level |
| Voiced F0 tail slope | Falling pitch = statement EOT |
| F0 tail mean / std | Final pitch context |
| F0 tail slope | Terminal intonation direction |
| Voicing ratio | Fraction of voiced frames |
| Last-run voiced length | Duration of final voiced stretch |
| Energy 10th / 90th percentile | Dynamic range |
| Energy range | Overall loudness variation |

### Files
- `train.py` — training entry point
- `../../eot_model.py` — feature extraction + model definition (shared with root)

## Usage

From the `speedrun/` root:
```bash
python approach1/train.py \
    --data_dirs eot_data/english eot_data/hindi \
    --out_model_dir .

python predict.py --data_dir eot_data/english --model_dir . --out predictions.csv
python score.py   --data_dir eot_data/english --pred predictions.csv
```

## Results
- 5-fold CV AUC: **0.594**
- Beaten by Approach 2 (CNN AUC 0.704 in CV)
