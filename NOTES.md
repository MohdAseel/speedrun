# NOTES

The current shell uses only the existing label schema and a deterministic fallback so the grader contract is fixed from the start. The eventual model should use only pre-pause audio features from the last voiced region, with no access to pause duration or post-pause audio. The hardest cases are short backchannels, filled pauses, and turns with weak final intonation. With one more day, I would add richer causal prosody features, tune the operating point on held-out turns, and compare a simple linear model against a small tree-based model.
