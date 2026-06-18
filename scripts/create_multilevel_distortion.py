"""
Phase 4.5 — Multi-level distortion: mild, medium, severe.
Each level stacks progressively stronger volume reduction, noise, low-pass filter,
and speed change on top of the previous one.
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_clean.csv"
TARGET_SR = 16_000
METADATA_OUT = ROOT / "data" / "metadata_multilevel_poc.csv"


# ── Distortion level definitions ───────────────────────────────────────────────

@dataclass
class DistortionLevel:
    name: str
    volume: float       # multiplier  (1.0 = original)
    noise_std: float    # Gaussian σ  (0.0 = silent)
    lowpass_hz: float   # cutoff Hz   (None = skip filter)
    speed: float        # rate        (1.0 = no change)


LEVELS: list[DistortionLevel] = [
    DistortionLevel(
        name="mild",
        volume=0.70,        # 70 % volume
        noise_std=0.003,    # barely audible hiss
        lowpass_hz=5000,    # wide-band — keeps most speech frequencies
        speed=1.03,         # almost imperceptible speed-up
    ),
    DistortionLevel(
        name="medium",
        volume=0.45,        # 45 % volume  (same as Phase 2 baseline)
        noise_std=0.008,
        lowpass_hz=3400,    # telephone band
        speed=1.07,
    ),
    DistortionLevel(
        name="severe",
        volume=0.20,        # 20 % — very quiet
        noise_std=0.030,    # clearly audible noise
        lowpass_hz=2000,    # narrow band, muffled
        speed=1.15,         # noticeably faster
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[multilevel_distortion] {msg}")


def check_inputs() -> None:
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_clean.csv not found: {METADATA_IN}")


def butter_lowpass_sos(cutoff_hz: float, sr: int, order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    return butter(order, cutoff_hz / nyq, btype="low", analog=False, output="sos")


def apply_level(audio: np.ndarray, sr: int, level: DistortionLevel) -> np.ndarray:
    """Apply one distortion level and return float32 audio normalised to [-1, 1]."""
    # 1. Volume
    out = audio * level.volume

    # 2. Gaussian noise (seeded per level name for reproducibility)
    rng = np.random.default_rng(seed=sum(ord(c) for c in level.name))
    out = out + rng.normal(0.0, level.noise_std, size=out.shape)

    # 3. Low-pass filter
    sos = butter_lowpass_sos(level.lowpass_hz, sr)
    out = sosfilt(sos, out).astype(np.float32)

    # 4. Speed change via resample
    out = librosa.resample(out, orig_sr=sr, target_sr=int(sr * level.speed))

    # 5. Normalise — keep 5 % headroom
    peak = np.max(np.abs(out))
    if peak > 0.0:
        out = out / peak * 0.95

    return out.astype(np.float32)


# ── Output folder map ──────────────────────────────────────────────────────────

def level_dir(level_name: str) -> Path:
    return ROOT / "data" / f"distorted_{level_name}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    check_inputs()

    for level in LEVELS:
        level_dir(level.name).mkdir(parents=True, exist_ok=True)
    log(f"Output folders ready: {[str(level_dir(l.name).relative_to(ROOT)) for l in LEVELS]}")

    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Loaded {len(rows)} entries from metadata_clean.csv")
    log("")

    out_rows: list[dict] = []
    skipped = 0

    for entry in rows:
        clean_id = entry["id"]          # e.g. clean_0001
        original_id = entry["original_id"]
        text = entry["text"]
        clean_path = ROOT / entry["clean_path"]

        suffix = clean_id.split("_")[-1]   # e.g. "0001"

        if not clean_path.exists():
            log(f"  [WARN] Missing clean wav — skipping {clean_id}: {clean_path}")
            skipped += 1
            continue

        # Load once, resample to TARGET_SR
        audio, _ = librosa.load(str(clean_path), sr=TARGET_SR, mono=True)

        level_paths: dict[str, str] = {}

        for level in LEVELS:
            out_name = f"distorted_{level.name}_{suffix}.wav"
            out_path = level_dir(level.name) / out_name

            distorted = apply_level(audio, TARGET_SR, level)
            sf.write(str(out_path), distorted, TARGET_SR, subtype="PCM_16")
            level_paths[level.name] = str(out_path.relative_to(ROOT))

        log(
            f"  {clean_id} → "
            + "  |  ".join(
                f"{l.name}: distorted_{l.name}_{suffix}.wav ({len(apply_level(audio, TARGET_SR, l)):,} samples)"
                for l in LEVELS
            )
        )

        out_rows.append({
            "id": clean_id,
            "original_id": original_id,
            "clean_path": entry["clean_path"],
            "distorted_mild_path": level_paths["mild"],
            "distorted_medium_path": level_paths["medium"],
            "distorted_severe_path": level_paths["severe"],
            "text": text,
        })

    # Write metadata
    fieldnames = [
        "id", "original_id", "clean_path",
        "distorted_mild_path", "distorted_medium_path", "distorted_severe_path",
        "text",
    ]
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log("")
    log(f"Wrote {len(out_rows)} entries to {METADATA_OUT}")
    if skipped:
        log(f"[WARN] Skipped {skipped} file(s) due to missing wavs")

    log("")
    log("Distortion level summary:")
    for level in LEVELS:
        log(
            f"  {level.name:8s}  volume={level.volume:.0%}  "
            f"noise_std={level.noise_std:.3f}  "
            f"lowpass={level.lowpass_hz:.0f}Hz  "
            f"speed=x{level.speed}"
        )
    log("Phase 4.5 complete.")


if __name__ == "__main__":
    main()
