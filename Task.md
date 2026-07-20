# End-of-Turn Detection

## The Problem

Every voice AI agent must constantly decide at every pause from the human user: is the human done talking, or just pausing?

If the AI agent answers too early, the agent talks over the human. If it answers too late, the conversation fills with silence.

Your task is to build a model from scratch that predicts the end of turn from real audio data. Details are below.

---

## The Data

You are given a folder `eot_data/` with two subfolders:

- `english/`
  - `audio/` – contains files named `en__NNN.wav` (16 kHz mono, one user turn per file)
  - `labels.csv`
- `hindi/`
  - `audio/` – contains files named `hi__NNN.wav`
  - `labels.csv`

Each WAV file represents one real user turn from a human‑to‑voice‑agent phone conversation (restaurant bookings, appointments, orders). Every silence pause of **≥ 100 ms** inside the turn is annotated in `labels.csv`.

### `labels.csv` Columns

| Column        | Meaning                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------- |
| `turn_id`     | which turn (matches the WAV filename)                                                    |
| `audio_file`  | relative path to the WAV file                                                            |
| `pause_index` | 0, 1, 2, … within the turn                                                               |
| `pause_start` | seconds – the moment speech stops                                                        |
| `pause_end`   | seconds – the moment speech resumes (or file ends)                                       |
| `label`       | `hold` = user continues after this pause, `eot` = this pause is the true end of the turn |

---

## Your Task

Write a model that, for **each pause**, outputs **p_eot**, the probability that the turn is over.

**Causality rule (hard requirement):**  
For a given pause, your features may use **only audio from time 0 up to `pause_start`** of that pause. Never use audio after the pause. A live agent cannot hear the future. Your feature code will be inspected to ensure compliance.

---

## Deliverables (all five required)

1. **`SUMMARY.html`** – a detailed HTML document describing your solution, results, graphs, and a summary of the MD files below. It should also explain what was done by you (human) and what was done by the coding agent, and why your solution beats the status quo.

2. **`predict.py`** – runs as:  
   `python predict.py --data_dir <folder> --out predictions.csv`  
   It must work on a folder with the same structure and label schema it has never seen before.  
   Predictions CSV columns: `turn_id`, `pause_index`, `p_eot`.

3. **`predictions.csv`** for **both** provided language folders.

4. **`RUNLOG.md`** – after every scoring run: the score, and 1‑2 lines on what you changed and why. This is graded.

5. **`NOTES.md`** – max 10 sentences: what signal your model uses, where it still fails, and what you would do with one more day.

---

## Scoring

Run:

```bash
python score.py --data_dir eot_data/english --pred predictions.csv
```

The scorer simulates a live agent with your scores and reports:

    Mean response delay (ms) at a false‑cutoff rate ≤ 5% – lower is better.

In plain words: tuned so that the agent interrupts users at most 5% of the time, how long does it keep users waiting after they have actually finished?

Your final grade uses a hidden test set (unseen turns, mostly Hindi) on our machine, plus your run log and a 5‑minute discussion of your model.

Reference point on this exact metric:

    Silence‑only baseline (given) ≈ 1600 ms

Rules

    Laptop CPU only. No GPUs, no cloud training.

    Allowed libraries: numpy, scipy, scikit‑learn, pandas, librosa, PyTorch.

    NOT allowed: any pretrained model or downloaded weights of any kind (no Whisper, wav2vec, Silero, webrtcvad, HuggingFace models, TTS/ASR APIs). No external datasets.

    AI coding assistants (Claude Code, Copilot, Codex, Cursor) are allowed. We calibrate for this: we know exactly what score they reach on their own.

    Your grading is based on what you add beyond what the coding agents do. If you just rely on them, you will not be able to get a good model working.

Suggested Path (not mandatory)

    0–10 min: run starter/baseline.py, score it, LISTEN to a few WAVs where hold pauses look long – understand why silence alone fails.

    10–35 min: extract prosodic features from the last ~1.5 s of speech before each pause (starter/features.py has audio loading and a pitch tracker). Train a small classifier. Score it.

    35–55 min: look at your worst errors. Fix what you see. Iterate.

    55–60 min: finalize predictions, create SUMMARY.html, RUNLOG.md, NOTES.md.
