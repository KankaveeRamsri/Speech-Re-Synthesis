"""
Phase 5 — Simple signal-processing enhancement baseline.
No AI/model training. Pipeline per file:
  1. Load as mono 16 kHz
  2. Spectral noise reduction (noisereduce)
  3. Band-pass filter  (80 Hz – 7500 Hz)  — removes sub-bass rumble & HF hiss
  4. Peak normalise    (target 0.95 FS)
"""

import csv
import sys
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_multilevel_poc.csv"
METADATA_OUT = ROOT / "data" / "metadata_enhanced_poc.csv"

TARGET_SR = 16_000

# Band-pass limits (Hz)
HP_CUTOFF = 80      # high-pass edge  — kill sub-bass rumble
LP_CUTOFF = 7500    # low-pass edge   — kill high-freq hiss above speech range
FILTER_ORDER = 5

# noisereduce settings
NR_PROP_DECREASE = 0.80   # how aggressively to reduce estimated noise (0–1)

LEVELS = ["medium", "severe"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[simple_enhancement] {msg}")


def check_inputs() -> None:
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_multilevel_poc.csv not found: {METADATA_IN}")


def out_dir(level: str) -> Path:
    return ROOT / "data" / f"enhanced_{level}"


def butter_bandpass_sos(lo: float, hi: float, sr: int, order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    return butter(order, [lo / nyq, hi / nyq], btype="band", analog=False, output="sos")


def enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    3-stage enhancement pipeline:
      1. Spectral noise reduction  (non-stationary mode)
      2. Band-pass filter          (80 Hz – 7500 Hz)
      3. Peak normalise            (headroom 5 %)
    """
    # Stage 1 — noise reduction
    # non-stationary=True suits time-varying noise introduced by our distortion
    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=NR_PROP_DECREASE,
    ).astype(np.float32)

    # Stage 2 — band-pass filter (remove sub-bass rumble + HF hiss)
    sos = butter_bandpass_sos(HP_CUTOFF, LP_CUTOFF, sr, order=FILTER_ORDER)
    filtered = sosfilt(sos, denoised).astype(np.float32)

    # Stage 3 — peak normalise
    peak = np.max(np.abs(filtered))
    if peak > 0.0:
        filtered = filtered / peak * 0.95

    return filtered


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    check_inputs()

    for level in LEVELS:
        out_dir(level).mkdir(parents=True, exist_ok=True)
    log(f"Output folders: {[str(out_dir(l).relative_to(ROOT)) for l in LEVELS]}")

    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Loaded {len(rows)} entries from metadata_multilevel_poc.csv")
    log("")

    out_rows: list[dict] = []
    skipped = 0

    for entry in rows:
        clean_id = entry["id"]          # e.g. clean_0001
        suffix = clean_id.split("_")[-1]   # e.g. "0001"
        enhanced_paths: dict[str, str] = {}

        row_ok = True
        for level in LEVELS:
            src_path = ROOT / entry[f"distorted_{level}_path"]

            if not src_path.exists():
                log(f"  [WARN] Missing distorted_{level} file — skipping {clean_id}: {src_path.name}")
                row_ok = False
                skipped += 1
                break

            # Load
            audio, _ = librosa.load(str(src_path), sr=TARGET_SR, mono=True)

            # Enhance
            enhanced = enhance(audio, TARGET_SR)

            # Save
            out_name = f"enhanced_{level}_{suffix}.wav"
            out_path = out_dir(level) / out_name
            sf.write(str(out_path), enhanced, TARGET_SR, subtype="PCM_16")
            enhanced_paths[level] = str(out_path.relative_to(ROOT))

            log(
                f"  [{level}] {clean_id} → {out_name}"
                f"  (in: {len(audio):,} samples → out: {len(enhanced):,} samples)"
            )

        if not row_ok:
            continue

        out_rows.append({
            "id": clean_id,
            "original_id": entry["original_id"],
            "clean_path": entry["clean_path"],
            "distorted_medium_path": entry["distorted_medium_path"],
            "distorted_severe_path": entry["distorted_severe_path"],
            "enhanced_medium_path": enhanced_paths["medium"],
            "enhanced_severe_path": enhanced_paths["severe"],
            "text": entry["text"],
        })

    log("")

    # Write metadata
    fieldnames = [
        "id", "original_id", "clean_path",
        "distorted_medium_path", "distorted_severe_path",
        "enhanced_medium_path", "enhanced_severe_path",
        "text",
    ]
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    log(f"Wrote {len(out_rows)} entries to {METADATA_OUT}")
    if skipped:
        log(f"[WARN] Skipped {skipped} file(s) due to missing inputs")
    log("")
    log("Enhancement pipeline summary:")
    log(f"  Stage 1 — Noise reduction  (noisereduce, prop_decrease={NR_PROP_DECREASE})")
    log(f"  Stage 2 — Band-pass filter ({HP_CUTOFF} Hz – {LP_CUTOFF} Hz, order {FILTER_ORDER})")
    log(f"  Stage 3 — Peak normalise   (target 0.95 FS)")
    log("Phase 5 complete.")


if __name__ == "__main__":
    main()
