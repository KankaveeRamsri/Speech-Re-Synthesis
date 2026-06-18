"""
Phase 2 — Create simulated distorted speech from clean audio.
Applies: volume reduction, Gaussian noise, low-pass filter, slight speed change.
"""

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_clean.csv"
CLEAN_DIR = ROOT / "data" / "clean"
DISTORTED_DIR = ROOT / "data" / "distorted"
METADATA_OUT = ROOT / "data" / "metadata_poc.csv"

TARGET_SR = 16_000          # resample everything to 16 kHz

VOLUME_FACTOR = 0.45        # reduce volume to 45 % of original
NOISE_STD = 0.008           # Gaussian noise amplitude (relative to float32 [-1, 1])
LOWPASS_CUTOFF_HZ = 3400    # telephone-like low-pass cutoff
SPEED_RATE = 1.07           # slight speed-up (1.0 = no change)


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[create_distortion] {msg}")


def check_inputs() -> None:
    if not CLEAN_DIR.exists():
        sys.exit(f"[ERROR] Clean audio folder not found: {CLEAN_DIR}")
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_clean.csv not found: {METADATA_IN}")


def butter_lowpass(cutoff_hz: float, sample_rate: int, order: int = 5) -> np.ndarray:
    nyq = sample_rate / 2.0
    normal_cutoff = cutoff_hz / nyq
    return butter(order, normal_cutoff, btype="low", analog=False, output="sos")


def apply_distortion(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply all distortion stages and return the result as float32 in [-1, 1]."""

    # 1. Volume reduction
    audio = audio * VOLUME_FACTOR

    # 2. Gaussian noise
    noise = np.random.default_rng(seed=42).normal(0.0, NOISE_STD, size=audio.shape)
    audio = audio + noise

    # 3. Low-pass filter (telephone-band simulation)
    sos = butter_lowpass(LOWPASS_CUTOFF_HZ, sr)
    audio = sosfilt(sos, audio).astype(np.float32)

    # 4. Slight speed change via resampling (no pitch shift)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=int(sr * SPEED_RATE))

    # 5. Normalize to [-1, 1] to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak * 0.95   # headroom of 5 %

    return audio.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    check_inputs()

    DISTORTED_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output folder ready: {DISTORTED_DIR}")

    # Read input metadata
    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Loaded {len(rows)} entries from metadata_clean.csv")

    out_rows: list[dict] = []
    skipped = 0

    for entry in rows:
        clean_id = entry["id"]           # e.g. clean_0001
        original_id = entry["original_id"]
        text = entry["text"]
        clean_path = ROOT / entry["clean_path"]

        # Derive distorted filename from the numeric suffix
        suffix = clean_id.split("_")[-1]  # e.g. "0001"
        dist_name = f"distorted_{suffix}.wav"
        dist_path = DISTORTED_DIR / dist_name

        if not clean_path.exists():
            log(f"  [WARN] Missing clean wav — skipping {clean_id}: {clean_path}")
            skipped += 1
            continue

        # Load and resample to TARGET_SR as mono float32
        audio, _ = librosa.load(str(clean_path), sr=TARGET_SR, mono=True)

        # Apply distortion pipeline
        distorted = apply_distortion(audio, TARGET_SR)

        # Save
        sf.write(str(dist_path), distorted, TARGET_SR, subtype="PCM_16")
        log(f"  {clean_id} → {dist_name}  (samples: {len(distorted):,})")

        out_rows.append(
            {
                "id": clean_id,
                "original_id": original_id,
                "clean_path": entry["clean_path"],
                "distorted_path": str(dist_path.relative_to(ROOT)),
                "text": text,
            }
        )

    # Write combined metadata
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "original_id", "clean_path", "distorted_path", "text"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    log(f"Wrote {len(out_rows)} entries to {METADATA_OUT}")
    if skipped:
        log(f"[WARN] Skipped {skipped} file(s) due to missing wavs")
    log("Phase 2 complete.")


if __name__ == "__main__":
    main()
