"""
Phase 9 — Convert predicted Mel-spectrograms to waveform audio (Griffin-Lim baseline).

Pipeline:
  predicted_mel (80, T)  [log scale, natural log]
  → exp() → linear mel (80, T)
  → mel_to_stft (librosa) → linear spectrogram
  → Griffin-Lim → waveform
  → save as .wav

Audio settings must match Phase 6 mel extraction:
  sample_rate = 16000
  n_mels      = 80
  hop_length  = 320
  win_length  = 1024
  n_fft       = 1024
"""

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
METADATA_IN      = ROOT / "data" / "metadata_predicted_mel.csv"
RECON_DIR        = ROOT / "data" / "reconstructed_audio"
METADATA_OUT     = ROOT / "data" / "metadata_reconstructed_audio.csv"

# ── Audio config (must match Phase 6 extract_wav2vec2_features.py) ─────────────
SAMPLE_RATE  = 16_000
N_MELS       = 80
HOP_LENGTH   = 320
WIN_LENGTH   = 1_024
N_FFT        = 1_024
N_ITER       = 60    # Griffin-Lim iterations — more = cleaner but slower


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[mel_to_waveform] {msg}")


# ── Conversion ─────────────────────────────────────────────────────────────────

def mel_to_audio(log_mel: np.ndarray) -> np.ndarray:
    """
    log_mel : (80, T)  — natural-log mel spectrogram (Phase 6 convention)
    returns : (N_samples,)  float32 waveform
    """
    # Undo log: recover linear-scale mel spectrogram
    mel_linear = np.exp(log_mel).astype(np.float32)   # (80, T)

    # Griffin-Lim via librosa helper (handles mel_to_stft + griffinlim internally)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_linear,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_iter=N_ITER,
        power=1.0,    # mel_to_audio expects amplitude, not power
    )
    return audio.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9 — predicted Mel → waveform via Griffin-Lim"
    )
    parser.add_argument("--metadata", type=Path, default=METADATA_IN,
                        help="Path to metadata_predicted_mel.csv")
    parser.add_argument("--n_iter", type=int, default=N_ITER,
                        help="Griffin-Lim iterations (default: 60)")
    args = parser.parse_args()

    # ── Validate ───────────────────────────────────────────────────────────────
    if not args.metadata.exists():
        sys.exit(f"[ERROR] Metadata file not found: {args.metadata}")

    RECON_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read metadata ──────────────────────────────────────────────────────────
    with args.metadata.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Metadata: {len(rows)} entries")
    log(f"Settings: sr={SAMPLE_RATE}  n_mels={N_MELS}  hop={HOP_LENGTH}  "
        f"win={WIN_LENGTH}  n_fft={N_FFT}  n_iter={args.n_iter}")
    log("")

    out_rows: list[dict] = []
    ok_count  = 0
    err_count = 0

    for idx, row in enumerate(rows, start=1):
        sample_id  = row["id"]
        pred_path  = ROOT / row["predicted_mel_path"]
        clean_path = row.get("clean_path", "")
        dist_path  = row.get("distorted_medium_path", "")
        text       = row.get("text", "")

        # Output filename: predicted_mel_0001 → reconstructed_0001.wav
        suffix   = sample_id.split("_")[-1]
        out_name = f"reconstructed_{suffix}.wav"
        out_path = RECON_DIR / out_name

        log(f"[{idx:>3}/{len(rows)}] {sample_id}")

        # ── Load predicted mel ─────────────────────────────────────────────────
        if not pred_path.exists():
            log(f"  [WARN] Predicted mel not found, skipping: {pred_path}")
            err_count += 1
            continue

        try:
            log_mel = np.load(str(pred_path))   # (80, T)
        except Exception as exc:
            log(f"  [WARN] Could not load {pred_path.name}: {exc}")
            err_count += 1
            continue

        if log_mel.ndim != 2 or log_mel.shape[0] != N_MELS:
            log(f"  [WARN] Unexpected mel shape {log_mel.shape}, expected (80, T) — skipping")
            err_count += 1
            continue

        log(f"  mel shape : {log_mel.shape}  min={log_mel.min():.3f}  max={log_mel.max():.3f}")

        # ── Convert to waveform ────────────────────────────────────────────────
        try:
            audio = mel_to_audio(log_mel)
        except Exception as exc:
            log(f"  [ERROR] Griffin-Lim failed: {exc}")
            err_count += 1
            continue

        duration_s = len(audio) / SAMPLE_RATE
        log(f"  audio     : {len(audio)} samples  ({duration_s:.2f}s)")

        # ── Save .wav ──────────────────────────────────────────────────────────
        try:
            sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        except Exception as exc:
            log(f"  [ERROR] Could not save wav: {exc}")
            err_count += 1
            continue

        log(f"  Saved     → {out_path.relative_to(ROOT)}")

        out_rows.append({
            "id":                    sample_id,
            "predicted_mel_path":    row["predicted_mel_path"],
            "reconstructed_audio_path": str(out_path.relative_to(ROOT)),
            "clean_path":            clean_path,
            "distorted_medium_path": dist_path,
            "text":                  text,
        })
        ok_count += 1
        log("")

    # ── Write metadata CSV ─────────────────────────────────────────────────────
    fieldnames = [
        "id", "predicted_mel_path", "reconstructed_audio_path",
        "clean_path", "distorted_medium_path", "text",
    ]
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log("=" * 56)
    log(f"  Processed : {ok_count} OK   {err_count} skipped")
    log(f"  Output dir: {RECON_DIR.relative_to(ROOT)}")
    log(f"  Metadata  : {METADATA_OUT.relative_to(ROOT)}")
    log("=" * 56)
    log("Phase 9 complete.")


if __name__ == "__main__":
    main()
