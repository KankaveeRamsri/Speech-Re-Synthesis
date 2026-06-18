"""
Phase 7 — Train a lightweight acoustic reconstruction model.

Maps:  Wav2Vec2 hidden states (T, 768) from distorted_medium audio
   →   clean log-Mel spectrogram frame  (T, 80)

Architecture choice — BiLSTM + linear projection:
  • BiLSTM captures temporal context in both directions
  • Single linear head per time-step maps 2*hidden → 80 mel bins
  • Frame-level L1 loss  (robust to outliers vs MSE)

Alignment: crop both sequences to min(T_w2v, T_mel) before training.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_features.csv"
MODELS_DIR = ROOT / "models"
CHECKPOINT = MODELS_DIR / "reconstruction_model.pt"
RESULTS_DIR = ROOT / "results"
TRAIN_LOG_CSV = RESULTS_DIR / "reconstruction_training_log.csv"
TRAIN_SUMMARY = RESULTS_DIR / "reconstruction_training_summary.txt"

# ── Training defaults ──────────────────────────────────────────────────────────
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[train_reconstruction] {msg}")


def load_pair(npz_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load and align one (wav2vec2, mel) pair.
    Returns (X, Y) both shape (T_min, dim) or None on error.
    """
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=True)
    x = data["wav2vec2_features"].astype(np.float32)   # (T_w2v, 768)
    mel = data["clean_mel"].astype(np.float32)          # (80, T_mel)
    y = mel.T                                            # → (T_mel, 80)

    t_min = min(x.shape[0], y.shape[0])
    return x[:t_min], y[:t_min]


# ── Dataset ────────────────────────────────────────────────────────────────────

class MelReconDataset(Dataset):
    """
    Returns one (X, Y) pair per sample where:
      X : (T, 768)  — Wav2Vec2 features from distorted_medium
      Y : (T,  80)  — log-Mel frames from clean audio (cropped to T)
    """

    def __init__(self, rows: list[dict]) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for entry in rows:
            pair = load_pair(ROOT / entry["feature_path"])
            if pair is None:
                log(f"  [WARN] Skipping missing file: {entry['feature_path']}")
                continue
            x, y = pair
            self.samples.append((torch.from_numpy(x), torch.from_numpy(y)))
        log(f"Dataset: {len(self.samples)} valid samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad variable-length sequences to the longest in the batch."""
    xs, ys = zip(*batch)
    x_lens = [x.shape[0] for x in xs]
    y_lens = [y.shape[0] for y in ys]

    T_max = max(x_lens)
    x_pad = torch.zeros(len(xs), T_max, xs[0].shape[1])
    y_pad = torch.zeros(len(ys), T_max, ys[0].shape[1])
    mask = torch.zeros(len(xs), T_max, dtype=torch.bool)   # True = valid frame

    for i, (x, y) in enumerate(zip(xs, ys)):
        t = x.shape[0]
        x_pad[i, :t] = x
        y_pad[i, :t] = y
        mask[i, :t] = True

    return x_pad, y_pad, mask


# ── Model ──────────────────────────────────────────────────────────────────────

class BiLSTMReconstructor(nn.Module):
    """
    BiLSTM encoder + linear frame-level projection.

    Input  : (B, T, 768)
    Output : (B, T,  80)

    BiLSTM sees the full sequence in both directions before projecting
    each frame to 80 mel-bin values.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_layers: int = 2, mel_bins: int = 80,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, mel_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, 768)
        out, _ = self.lstm(x)          # (B, T, 2*hidden)
        return self.proj(out)          # (B, T, 80)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> list[dict]:
    model.train()
    log_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_frames = 0

        for x_pad, y_pad, mask in loader:
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(x_pad)                      # (B, T, 80)

            # Only compute loss on non-padded frames
            pred_valid = pred[mask]                  # (N_valid_frames, 80)
            y_valid = y_pad[mask]
            loss = criterion(pred_valid, y_valid)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * pred_valid.shape[0]
            n_frames += pred_valid.shape[0]

        avg_loss = epoch_loss / max(n_frames, 1)
        log_rows.append({"epoch": epoch, "avg_loss": round(avg_loss, 6)})

        if epoch % 5 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d}/{epochs}  avg_loss={avg_loss:.6f}")

    return log_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 — reconstruction model training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_features.csv not found: {METADATA_IN}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Metadata: {len(rows)} entries")

    dataset = MelReconDataset(rows)
    if len(dataset) == 0:
        sys.exit("[ERROR] No valid samples loaded — check feature files.")

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch, len(dataset)),
        shuffle=True,
        collate_fn=collate_pad,
        drop_last=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = BiLSTMReconstructor(
        input_dim=768,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        mel_bins=80,
        dropout=DEFAULT_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model: BiLSTMReconstructor  params={n_params:,}")
    log(f"Hyperparams: epochs={args.epochs}  lr={args.lr}  hidden={args.hidden}  layers={args.layers}  batch={args.batch}")
    log("")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # ── Train ──────────────────────────────────────────────────────────────────
    log(f"Training for {args.epochs} epochs ...")
    log_rows = train(model, loader, optimizer, criterion, device, args.epochs)

    final_loss = log_rows[-1]["avg_loss"]
    first_loss = log_rows[0]["avg_loss"]
    improvement = first_loss - final_loss

    log("")
    log("=" * 56)
    log(f"  Samples          : {len(dataset)}")
    log(f"  Epochs           : {args.epochs}")
    log(f"  Initial loss     : {first_loss:.6f}")
    log(f"  Final loss       : {final_loss:.6f}")
    log(f"  Improvement      : {improvement:.6f}  ({improvement/first_loss*100:.1f}%)")
    log("=" * 56)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": {
            "input_dim": 768,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "mel_bins": 80,
            "dropout": DEFAULT_DROPOUT,
        },
        "final_loss": final_loss,
        "epochs_trained": args.epochs,
    }, str(CHECKPOINT))
    log(f"Checkpoint saved → {CHECKPOINT.relative_to(ROOT)}")

    # ── Write training log CSV ─────────────────────────────────────────────────
    with TRAIN_LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "avg_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    log(f"Training log  → {TRAIN_LOG_CSV.relative_to(ROOT)}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    summary_lines = [
        "Speech Re-Synthesis POC — Reconstruction Model Training Summary",
        "=" * 62,
        f"Model architecture    : BiLSTMReconstructor",
        f"Parameters            : {n_params:,}",
        f"Input dim             : 768  (Wav2Vec2 hidden size)",
        f"Output dim            : 80   (Mel bins)",
        f"LSTM hidden           : {args.hidden} x {args.layers} layers  (bidirectional)",
        f"Loss function         : L1Loss (frame-level)",
        f"Optimizer             : Adam  lr={args.lr}",
        f"Batch size            : {args.batch}",
        f"Training samples      : {len(dataset)}",
        f"Epochs                : {args.epochs}",
        "",
        f"Initial avg loss      : {first_loss:.6f}",
        f"Final avg loss        : {final_loss:.6f}",
        f"Total improvement     : {improvement:.6f}  ({improvement/first_loss*100:.1f}%)",
        "",
        f"Checkpoint            : {CHECKPOINT.relative_to(ROOT)}",
        f"Training log CSV      : {TRAIN_LOG_CSV.relative_to(ROOT)}",
    ]
    TRAIN_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary       → {TRAIN_SUMMARY.relative_to(ROOT)}")
    log("Phase 7 complete.")


if __name__ == "__main__":
    main()
