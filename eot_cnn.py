"""Approach 2+: Hybrid CNN-RF feature fusion for EOT detection.

Upgrade over pure CNN approach: CNN Mel-spectrogram embeddings are
concatenated with handcrafted prosodic features from eot_model.py and
fed into a small MLP + optional RF head. This gives the network both
structured priors (pitch slope, energy decay) and learned spectral
texture simultaneously.

Usage
-----
    python train_approach2.py --data_dirs eot_data/english eot_data/hindi \
        --out_model_dir . --epochs 50 --hybrid

The --hybrid flag activates feature fusion mode (recommended).
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from features import load_wav, speech_before
from eot_model import extract_features as rf_extract_features

CNN_FILENAME = "eot_cnn.pt"

# ── Spectrogram hyper-params ──────────────────────────────────────────────────
N_MELS = 40
HOP_LEN = 160    # 10 ms at 16 kHz
N_FFT = 400      # 25 ms window
T_FRAMES = 150   # fixed width = 1.5 s at 10 ms hop (pad/truncate)
SR_EXPECTED = 16_000
RF_FEAT_DIM = 16  # from eot_model.extract_features


# ── Feature extraction ────────────────────────────────────────────────────────

def _mel_spectrogram(seg: np.ndarray, sr: int) -> np.ndarray:
    """Return a (N_MELS, T_FRAMES) log-Mel spectrogram, normalised to mu=0 sigma=1."""
    if sr != SR_EXPECTED:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=SR_EXPECTED)
        sr = SR_EXPECTED

    S = librosa.feature.melspectrogram(
        y=seg, sr=sr,
        n_mels=N_MELS, hop_length=HOP_LEN,
        n_fft=N_FFT, fmin=60.0, fmax=8000.0,
    )
    S_db = librosa.power_to_db(S + 1e-9, ref=1.0)  # (N_MELS, T_actual)

    # Pad or truncate time axis to T_FRAMES (keep the most recent frames)
    t = S_db.shape[1]
    if t >= T_FRAMES:
        S_db = S_db[:, -T_FRAMES:]
    else:
        pad = T_FRAMES - t
        S_db = np.pad(S_db, ((0, 0), (pad, 0)), mode="constant")

    mu = S_db.mean()
    sigma = S_db.std() + 1e-6
    return ((S_db - mu) / sigma).astype(np.float32)


def mel_features(x: np.ndarray, sr: int, pause_start: float) -> np.ndarray:
    """Causal Mel-spectrogram from the last 1.5 s before pause_start.

    Returns (N_MELS, T_FRAMES) float32 array.
    Causality: only uses audio[0 : pause_start]. pause_end is never touched.
    """
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 20:
        return np.zeros((N_MELS, T_FRAMES), dtype=np.float32)
    return _mel_spectrogram(seg, sr)


# ── Model definitions ─────────────────────────────────────────────────────────

class EOTConvNet(nn.Module):
    """~14k-parameter 2-D CNN for EOT binary classification.

    Input shape:  (B, 1, N_MELS=40, T_FRAMES=150)
    Output shape: (B, EMB_DIM) embedding  or  (B,) logit when head=True
    """
    EMB_DIM = 32

    def __init__(self, return_embedding: bool = False):
        super().__init__()
        self.return_embedding = return_embedding
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),           # -> (B, 16, 20, 75)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),           # -> (B, 32, 10, 37)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # -> (B, 32, 1, 1)
        )
        self.flatten = nn.Sequential(
            nn.Flatten(),              # -> (B, 32)
        )
        # Classification head (only used in standalone mode)
        self.head = nn.Sequential(
            nn.Linear(self.EMB_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        emb = self.flatten(x)          # (B, 32)
        if self.return_embedding:
            return emb
        return self.head(emb).squeeze(1)   # (B,)


class HybridEOTNet(nn.Module):
    """Hybrid: CNN embedding + RF prosodic features -> MLP classifier.

    Total parameters: ~18k. Combines learned spectral texture with
    handcrafted pitch/energy features for robust generalisation on small data.
    """

    def __init__(self, rf_dim: int = RF_FEAT_DIM):
        super().__init__()
        self.cnn = EOTConvNet(return_embedding=True)   # outputs (B, 32)
        # RF-feature branch: light normalisation + projection
        self.rf_proj = nn.Sequential(
            nn.Linear(rf_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )

    def forward(
        self, spec: torch.Tensor, rf: torch.Tensor
    ) -> torch.Tensor:
        """spec: (B,1,40,150)  rf: (B, RF_FEAT_DIM)"""
        cnn_emb = self.cnn(spec)           # (B, 32)
        rf_emb  = self.rf_proj(rf)         # (B, 32)
        fused   = torch.cat([cnn_emb, rf_emb], dim=1)  # (B, 64)
        return self.fusion(fused).squeeze(1)            # (B,)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PauseDataset(Dataset):
    """Maps label rows to (mel_spec_tensor, rf_feat_tensor, label_float, turn_id)."""

    def __init__(self, rows: list, augment: bool = False):
        self.rows = rows
        self.augment = augment
        self._cache: dict = {}

    def _load(self, path: str) -> Tuple[np.ndarray, int]:
        if path not in self._cache:
            self._cache[path] = load_wav(path)
        return self._cache[path]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x, sr = self._load(row["audio_path"])
        spec = mel_features(x, sr, row["pause_start"])
        rf   = row["rf_feats"]   # already computed at load time

        if self.augment:
            # Amplitude jitter +/-3 dB in log-mel domain
            gain_db = random.uniform(-3.0, 3.0)
            spec = spec + gain_db
            # Time shift +/-5 frames (~50 ms)
            shift = random.randint(-5, 5)
            if shift > 0:
                spec = np.concatenate(
                    [np.zeros((N_MELS, shift), dtype=np.float32), spec[:, :-shift]],
                    axis=1,
                )
            elif shift < 0:
                spec = np.concatenate(
                    [spec[:, -shift:], np.zeros((N_MELS, -shift), dtype=np.float32)],
                    axis=1,
                )
            # Slight RF feature noise (5% gaussian)
            rf = rf + np.random.randn(*rf.shape).astype(np.float32) * 0.05 * (np.abs(rf) + 1e-6)

        spec_t  = torch.from_numpy(spec).unsqueeze(0)  # (1, N_MELS, T_FRAMES)
        rf_t    = torch.from_numpy(rf.astype(np.float32))
        label_t = torch.tensor(row["label"], dtype=torch.float32)
        return spec_t, rf_t, label_t, row["turn_id"]


def _load_rows(data_dirs: List[str]) -> list:
    """Load labels + pre-compute RF features (cached per WAV)."""
    rows = []
    audio_cache: dict = {}
    for d in data_dirs:
        labels_path = os.path.join(d, "labels.csv")
        with open(labels_path) as f:
            for r in csv.DictReader(f):
                ap = os.path.join(d, r["audio_file"])
                if ap not in audio_cache:
                    audio_cache[ap] = load_wav(ap)
                x, sr = audio_cache[ap]
                rf_feats = rf_extract_features(x, sr, float(r["pause_start"]))
                rows.append({
                    "turn_id":     r["turn_id"],
                    "audio_path":  ap,
                    "pause_start": float(r["pause_start"]),
                    "label":       1 if r["label"] == "eot" else 0,
                    "rf_feats":    rf_feats,
                })
    return rows


def _group_split(
    rows: list, val_frac: float = 0.20, seed: int = 0
) -> Tuple[list, list]:
    """Split by turn_id to avoid data leakage between train and val."""
    rng = random.Random(seed)
    turns = list({r["turn_id"] for r in rows})
    rng.shuffle(turns)
    n_val = max(1, int(len(turns) * val_frac))
    val_turns = set(turns[:n_val])
    train_rows = [r for r in rows if r["turn_id"] not in val_turns]
    val_rows   = [r for r in rows if r["turn_id"] in val_turns]
    return train_rows, val_rows


# ── Training ──────────────────────────────────────────────────────────────────

def train_cnn(
    data_dirs: List[str],
    out_dir: str = ".",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    hybrid: bool = True,
) -> float:
    """Train HybridEOTNet (or standalone EOTConvNet) and save the best checkpoint.

    Returns the best validation AUC.
    Saves to out_dir/eot_cnn.pt.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Pre-loading audio and computing features...")
    rows = _load_rows(data_dirs)
    train_rows, val_rows = _group_split(rows, val_frac=0.20, seed=seed)
    n_train_turns = len({r["turn_id"] for r in train_rows})
    n_val_turns   = len({r["turn_id"] for r in val_rows})
    print(f"Train: {len(train_rows)} pauses from {n_train_turns} turns")
    print(f"Val  : {len(val_rows)} pauses from {n_val_turns} turns")

    train_ds = PauseDataset(train_rows, augment=True)
    val_ds   = PauseDataset(val_rows,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Imbalance weight
    n_pos = sum(r["label"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    print(f"pos_weight = {pos_weight.item():.2f}  (hold/eot ratio in train)")

    model: nn.Module
    if hybrid:
        model = HybridEOTNet(rf_dim=RF_FEAT_DIM)
        print("Mode: HYBRID (CNN + RF feature fusion)")
    else:
        model = EOTConvNet(return_embedding=False)
        print("Mode: CNN only")
    print(f"Model parameters: {_count_params(model):,}")

    # Label smoothing via BCE with logits
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_auc   = 0.0
    best_state = None
    out_path   = Path(out_dir) / CNN_FILENAME

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for specs, rfs, labels, _ in train_dl:
            optimizer.zero_grad()
            if hybrid:
                logits = model(specs, rfs)
            else:
                logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_rows)

        # -- validate --
        model.eval()
        all_probs:  List[float] = []
        all_labels: List[int]   = []
        with torch.no_grad():
            for specs, rfs, labels, _ in val_dl:
                if hybrid:
                    logits = model(specs, rfs)
                else:
                    logits = model(specs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())

        has_both = len(set(all_labels)) > 1
        val_auc = roc_auc_score(all_labels, all_probs) if has_both else 0.5
        scheduler.step()

        marker = " <-- best" if val_auc > best_auc else ""
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={train_loss:.4f}  val_AUC={val_auc:.4f}{marker}"
        )

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save: store both state_dict AND hybrid flag so load_cnn knows how to reconstruct
    save_payload = {
        "state_dict": model.state_dict(),
        "hybrid": hybrid,
        "rf_feat_dim": RF_FEAT_DIM,
    }
    torch.save(save_payload, out_path)
    print(f"\nBest val AUC: {best_auc:.4f}  ->  saved to {out_path}")
    return best_auc


# ── Inference ─────────────────────────────────────────────────────────────────

def load_cnn(model_dir: str | os.PathLike | None) -> "nn.Module | None":
    """Load a trained EOTConvNet / HybridEOTNet from <model_dir>/eot_cnn.pt.

    Returns None if the file does not exist.
    """
    if model_dir is None:
        return None
    p = Path(model_dir) / CNN_FILENAME
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        hybrid = payload.get("hybrid", False)
        rf_dim = payload.get("rf_feat_dim", RF_FEAT_DIM)
        if hybrid:
            model: nn.Module = HybridEOTNet(rf_dim=rf_dim)
        else:
            model = EOTConvNet(return_embedding=False)
        model.load_state_dict(payload["state_dict"])
    else:
        # Legacy: plain state_dict from first version
        model = EOTConvNet(return_embedding=False)
        model.load_state_dict(payload)
    model.eval()
    return model


def predict_cnn(
    model: nn.Module,
    x: np.ndarray,
    sr: int,
    pause_start: float,
) -> float:
    """Return p_eot in [0, 1] for a single pause.

    Automatically uses RF features when model is HybridEOTNet.
    """
    spec = mel_features(x, sr, pause_start)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, 40, 150)

    with torch.no_grad():
        if isinstance(model, HybridEOTNet):
            from eot_model import extract_features as _rf
            rf_feat = _rf(x, sr, pause_start).astype(np.float32)
            rf_t = torch.from_numpy(rf_feat).unsqueeze(0)       # (1, 16)
            logit = model(spec_t, rf_t)
        else:
            logit = model(spec_t)

    return float(torch.sigmoid(logit).item())
