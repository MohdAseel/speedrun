# RUNLOG

- 2026-07-20: English baseline shell scored 1600 ms mean response delay at 0.0% interrupted turns. I kept the inference contract fixed and validated the submission CLI end-to-end before training.
- 2026-07-21: Added a shared causal feature/model scaffold in eot_model.py and wired predict.py to load a saved artifact when available. Re-validated the scorer; the fallback still scores 1600 ms until a trained model is saved.
