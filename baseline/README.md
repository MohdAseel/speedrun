# Baseline — Silence-Only EOT Detector

## What It Does

The reference starting point: predicts `p_eot = 1.0` for every pause (fire immediately on every silence). The scorer then sweeps thresholds and finds the best operating point, which lands at **1600 ms mean response delay at 0% interrupted turns**.

This is the floor. Everything above beats it.

## Usage

From the `speedrun/` root:
```bash
python baseline/run.py --data_dir eot_data/english --out predictions_baseline.csv
python score.py        --data_dir eot_data/english --pred predictions_baseline.csv
```

**Expected output:**
```
turns=100  pauses=248  AUC=0.514
BEST @ <= 5% interrupted turns:
  mean response delay : 1600 ms
  interrupted turns   : 0.0%
```

## Why It Scores 1600ms

The scorer waits up to 1.6 s (the `TIMEOUT_S` constant) before forcing a response on true EOT pauses. When `p_eot = 1.0` for every pause including hold pauses, the agent fires immediately — interrupting the user. To stay within the 5% false-cutoff budget, the scorer must choose a high delay, landing at the maximum 1600 ms timeout.
