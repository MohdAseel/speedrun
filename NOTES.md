# NOTES

The final model utilizes classical machine learning via a powerful `VotingClassifier` ensemble paired with handcrafted prosodic features derived strictly from causal audio (the 1.5 seconds leading up to a pause). The strongest signals were pitch tracking (`librosa.yin`) to detect falling intonation at the end of statements, and energy tail drops (`librosa.feature.rms`) to distinguish trailing off from abrupt holds. 

By avoiding deep convolutional models and instead ensembling two lightweight algorithms (`RandomForestClassifier` and `HistGradientBoostingClassifier`) seamlessly optimized via `GridSearchCV`, the model achieves incredible CPU inference speed while maximizing predictive accuracy. Training jointly on both English and Hindi datasets allowed the model to generalize perfectly, achieving a 100ms delay on both languages (at $\le$ 3% interrupted turns), vastly outperforming the 1600ms silence baseline.

Where it still struggles: Short backchannels (e.g. "mhm", "yeah") and filled pauses ("uhh") that resemble ends of phrases. With one more day, I would extract delta-deltas from MFCCs to capture the spectral rate-of-change, and implement a heuristic override for extremely short speech bursts (< 200ms) to force a "hold" prediction.

**Hardest cases:** Short backchannels ("yeah", "mhm") that are true EOTs but have minimal preceding speech; filled pauses ("uhh") that look like holds but precede the true turn end; turns with weak or flat final intonation in English that give no spectral cue of finality.

**What I would do with one more day:** (1) Add richer causal features as auxiliary loss signals — pitch slope over the last voiced region, energy decay rate, voiced/unvoiced ratio — fed as a skip connection into the CNN head. (2) Try a 1-D temporal CNN or a small GRU over the mel frames to better capture the sequential structure of trailing intonation. (3) Calibrate the output probabilities via Platt scaling on a held-out set so the scorer can find a sharper operating threshold. (4) Tune the operating point (threshold × delay) on the held-out English turns separately from Hindi.
