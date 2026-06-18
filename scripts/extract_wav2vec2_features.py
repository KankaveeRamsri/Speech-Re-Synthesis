"""
Phase 6 — Extract Wav2Vec2 features from distorted_medium audio
and clean Mel-spectrogram targets from clean audio.

Outputs per sample:
  data/features_wav2vec2/<id>.npz  — wav2vec2_features, clean_mel, id, text, paths
  data/metadata_features.csv       — index with shapes for downstream training
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# silence the expected lm_head / masked_spec_embed load warnings
warnings.filterwarnings("ignore", message=".*lm_head.*")
warnings.filterwarnings("ignore", message=".*masked_spec_embed.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_multilevel_poc.csv"
FEATURES_DIR = ROOT / "data" / "features_wav2vec2"
METADATA_OUT = ROOT / "data" / "metadata_features.csv"

# ── Audio / feature config ─────────────────────────────────────────────────────
TARGET_SR = 16_000
N_MELS = 80
HOP_LENGTH = 320
WIN_LENGTH = 1024

MODEL_ID = "facebook/wav2vec2-base-960h"
DEFAULT_N = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[extract_features] {msg}")


def check_inputs() -> None:
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_multilevel_poc.csv not found: {METADATA_IN}")


def load_mono_16k(path: Path) -> np.ndarray:
    """Load audio as float32 mono array at 16 kHz."""
    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def extract_wav2vec2(
    audio: np.ndarray,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    """
    Return Wav2Vec2 hidden states as float32 numpy array (T, 768).
    The processor normalises raw waveform; we take last_hidden_state.
    """
    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        out = model(input_values)

    # (1, T, 768) → (T, 768)
    return out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """
    Return log-Mel spectrogram as float32 numpy array (n_mels, T_mel).
    Uses librosa melspectrogram then log-scale with small floor to avoid -inf.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float32)  # (80, T_mel)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 — Wav2Vec2 feature extraction")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of samples to process (default: {DEFAULT_N}, 0 = all)")
    args = parser.parse_args()

    check_inputs()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Loading model: {MODEL_ID} ...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device)
    model.eval()
    log("Model ready.")
    log("")

    # ── Load metadata ──────────────────────────────────────────────────────────
    with METADATA_IN.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[: args.n] if args.n > 0 else all_rows
    log(f"Processing {len(rows)} / {len(all_rows)} samples")
    log("")

    out_meta: list[dict] = []
    skipped = 0

    for i, entry in enumerate(rows, start=1):
        row_id = entry["id"]
        text = entry["text"]
        clean_path = ROOT / entry["clean_path"]
        dist_path = ROOT / entry["distorted_medium_path"]
        npz_path = FEATURES_DIR / f"{row_id}.npz"

        log(f"[{i}/{len(rows)}] {row_id}")

        # Check files exist
        missing = [p for p in (clean_path, dist_path) if not p.exists()]
        if missing:
            for p in missing:
                log(f"  [WARN] Missing: {p.name}")
            skipped += 1
            continue

        # Load audio
        clean_audio = load_mono_16k(clean_path)
        dist_audio = load_mono_16k(dist_path)
        log(f"  clean  : {len(clean_audio):,} samples ({len(clean_audio)/TARGET_SR:.2f}s)")
        log(f"  distort: {len(dist_audio):,} samples ({len(dist_audio)/TARGET_SR:.2f}s)")

        # Extract Wav2Vec2 features from distorted_medium
        w2v_features = extract_wav2vec2(dist_audio, processor, model, device)
        log(f"  wav2vec2 features : {w2v_features.shape}  (T_w2v x 768)")

        # Compute clean log-Mel spectrogram
        clean_mel = compute_mel(clean_audio)
        log(f"  clean mel         : {clean_mel.shape}  (80 x T_mel)")

        # Save .npz
        np.savez_compressed(
            str(npz_path),
            wav2vec2_features=w2v_features,
            clean_mel=clean_mel,
            id=row_id,
            text=text,
            clean_path=str(clean_path.relative_to(ROOT)),
            distorted_medium_path=str(dist_path.relative_to(ROOT)),
        )
        log(f"  saved → {npz_path.relative_to(ROOT)}")
        log("")

        out_meta.append({
            "id": row_id,
            "feature_path": str(npz_path.relative_to(ROOT)),
            "clean_path": entry["clean_path"],
            "distorted_medium_path": entry["distorted_medium_path"],
            "text": text,
            "wav2vec2_shape": f"{w2v_features.shape[0]}x{w2v_features.shape[1]}",
            "mel_shape": f"{clean_mel.shape[0]}x{clean_mel.shape[1]}",
        })

    # ── Write metadata CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "id", "feature_path", "clean_path", "distorted_medium_path",
        "text", "wav2vec2_shape", "mel_shape",
    ]
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_meta)

    log("=" * 56)
    log(f"  Processed         : {len(out_meta)}")
    log(f"  Skipped           : {skipped}")
    log(f"  Features folder   : {FEATURES_DIR.relative_to(ROOT)}")
    log(f"  Metadata CSV      : {METADATA_OUT.relative_to(ROOT)}")
    if out_meta:
        log(f"  Wav2Vec2 dim      : 768 hidden units per frame")
        log(f"  Mel bins          : {N_MELS}  hop={HOP_LENGTH}  win={WIN_LENGTH}")
    log("=" * 56)
    log("Phase 6 complete.")


if __name__ == "__main__":
    main()
