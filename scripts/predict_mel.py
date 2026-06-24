"""
Phase 8 — Run inference with the trained BiLSTM reconstruction model.

Reads   : data/metadata_features.csv  +  data/features_wav2vec2/*.npz
Loads   : models/reconstruction_model.pt
Outputs : data/predicted_mel/predicted_mel_XXXX.npy   (80, T)
          data/metadata_predicted_mel.csv
          results/mel_preview_0001..0003.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN   = ROOT / "data" / "metadata_features.csv"
CHECKPOINT    = ROOT / "models" / "reconstruction_model.pt"
PRED_DIR      = ROOT / "data" / "predicted_mel"
METADATA_OUT  = ROOT / "data" / "metadata_predicted_mel.csv"
RESULTS_DIR   = ROOT / "results"

PREVIEW_COUNT = 3   # save visual comparisons for the first N samples


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[predict_mel] {msg}")


# ── Model (must match train_reconstruction_model.py exactly) ───────────────────

class BiLSTMReconstructor(nn.Module):
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
        out, _ = self.lstm(x)
        return self.proj(out)


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    hp = ckpt.get("hyperparams", {})
    model = BiLSTMReconstructor(
        input_dim  = hp.get("input_dim",  768),
        hidden_dim = hp.get("hidden_dim", 256),
        num_layers = hp.get("num_layers", 2),
        mel_bins   = hp.get("mel_bins",   80),
        dropout    = hp.get("dropout",    0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    trained_epochs = ckpt.get("epochs_trained", "?")
    final_loss     = ckpt.get("final_loss", float("nan"))
    log(f"Checkpoint loaded  epochs={trained_epochs}  final_loss={final_loss:.6f}")
    return model


# ── Inference for one sample ───────────────────────────────────────────────────

@torch.no_grad()
def predict_one(model: nn.Module, wav2vec2_feat: np.ndarray,
                device: torch.device) -> np.ndarray:
    """
    wav2vec2_feat : (T, 768)
    returns       : predicted mel (80, T)
    """
    x = torch.from_numpy(wav2vec2_feat.astype(np.float32)).unsqueeze(0).to(device)
    pred = model(x)          # (1, T, 80)
    pred_np = pred.squeeze(0).cpu().numpy()   # (T, 80)
    return pred_np.T                           # (80, T)


# ── Visualization ──────────────────────────────────────────────────────────────

def save_preview(sample_id: str, clean_mel: np.ndarray, pred_mel: np.ndarray,
                 out_path: Path) -> None:
    """
    Save a side-by-side comparison of clean vs predicted mel-spectrogram.
    Both inputs expected as (80, T); cropped to shorter for alignment.
    """
    t_min = min(clean_mel.shape[1], pred_mel.shape[1])
    clean  = clean_mel[:, :t_min]
    pred   = pred_mel[:, :t_min]

    vmin = min(clean.min(), pred.min())
    vmax = max(clean.max(), pred.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Mel Spectrogram Comparison — {sample_id}", fontsize=12)

    axes[0].imshow(clean, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].set_title("Clean Mel (target)")
    axes[0].set_ylabel("Mel bin")

    im = axes[1].imshow(pred, aspect="auto", origin="lower",
                        vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].set_title("Predicted Mel (model output)")
    axes[1].set_ylabel("Mel bin")
    axes[1].set_xlabel("Frame")

    fig.colorbar(im, ax=axes, location="right", shrink=0.8, label="Log energy")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Preview saved → {out_path.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 — predict Mel from Wav2Vec2 features")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="Path to reconstruction_model.pt")
    parser.add_argument("--metadata",   type=Path, default=METADATA_IN,
                        help="Path to metadata_features.csv")
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not args.metadata.exists():
        sys.exit(f"[ERROR] Metadata file not found: {args.metadata}")
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    # ── Setup ──────────────────────────────────────────────────────────────────
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    # ── Read metadata ──────────────────────────────────────────────────────────
    with args.metadata.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Metadata: {len(rows)} entries found")
    log("")

    out_rows: list[dict] = []
    ok_count = 0
    err_count = 0

    for idx, row in enumerate(rows, start=1):
        sample_id   = row["id"]
        feat_path   = ROOT / row["feature_path"]
        clean_path  = row.get("clean_path", "")
        dist_path   = row.get("distorted_medium_path", "")
        text        = row.get("text", "")

        # Extract numeric suffix, e.g. "clean_0001" → "0001"
        suffix = sample_id.split("_")[-1]
        out_name = f"predicted_mel_{suffix}.npy"
        pred_path = PRED_DIR / out_name

        log(f"[{idx:>3}/{len(rows)}] {sample_id}")

        # ── Load features ──────────────────────────────────────────────────────
        if not feat_path.exists():
            log(f"  [WARN] Feature file missing, skipping: {feat_path}")
            err_count += 1
            continue

        try:
            npz = np.load(str(feat_path), allow_pickle=True)
        except Exception as exc:
            log(f"  [WARN] Could not load {feat_path.name}: {exc}")
            err_count += 1
            continue

        if "wav2vec2_features" not in npz:
            log(f"  [WARN] 'wav2vec2_features' key missing in {feat_path.name}, skipping")
            err_count += 1
            continue

        wav2vec2 = npz["wav2vec2_features"]   # (T, 768)
        clean_mel_raw = npz.get("clean_mel")  # (80, T_mel) — may be None

        log(f"  wav2vec2 shape: {wav2vec2.shape}")

        # ── Run inference ──────────────────────────────────────────────────────
        try:
            pred_mel = predict_one(model, wav2vec2, device)   # (80, T)
        except Exception as exc:
            log(f"  [ERROR] Inference failed: {exc}")
            err_count += 1
            continue

        log(f"  predicted mel shape: {pred_mel.shape}")

        # ── Save .npy ──────────────────────────────────────────────────────────
        np.save(str(pred_path), pred_mel)
        log(f"  Saved → {pred_path.relative_to(ROOT)}")

        # ── Save preview for first PREVIEW_COUNT samples ───────────────────────
        if idx <= PREVIEW_COUNT and clean_mel_raw is not None:
            preview_path = RESULTS_DIR / f"mel_preview_{suffix}.png"
            try:
                save_preview(sample_id, clean_mel_raw, pred_mel, preview_path)
            except Exception as exc:
                log(f"  [WARN] Preview failed: {exc}")
        elif idx <= PREVIEW_COUNT:
            log(f"  [INFO] No clean_mel in npz — skipping preview for {sample_id}")

        out_rows.append({
            "id":                   sample_id,
            "feature_path":         row["feature_path"],
            "predicted_mel_path":   str(pred_path.relative_to(ROOT)),
            "clean_path":           clean_path,
            "distorted_medium_path": dist_path,
            "text":                 text,
            "predicted_mel_shape":  f"{pred_mel.shape[0]}x{pred_mel.shape[1]}",
        })
        ok_count += 1
        log("")

    # ── Write metadata CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "id", "feature_path", "predicted_mel_path",
        "clean_path", "distorted_medium_path", "text", "predicted_mel_shape",
    ]
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log("=" * 56)
    log(f"  Processed : {ok_count} OK   {err_count} skipped")
    log(f"  Output dir: {PRED_DIR.relative_to(ROOT)}")
    log(f"  Metadata  : {METADATA_OUT.relative_to(ROOT)}")
    log(f"  Previews  : results/mel_preview_XXXX.png (first {PREVIEW_COUNT})")
    log("=" * 56)
    log("Phase 8 complete.")


if __name__ == "__main__":
    main()
