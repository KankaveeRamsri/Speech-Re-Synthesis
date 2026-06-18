"""
Phase 1 — Prepare clean audio subset from LJ Speech.
Copies the first 50 samples into data/clean/ and writes data/metadata_clean.csv.
"""

import csv
import shutil
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "LJSpeech-1.1 3"
METADATA_IN = DATASET_DIR / "metadata.csv"
WAVS_DIR = DATASET_DIR / "wavs"
CLEAN_DIR = ROOT / "data" / "clean"
METADATA_OUT = ROOT / "data" / "metadata_clean.csv"

NUM_SAMPLES = 50


def log(msg: str) -> None:
    print(f"[prepare_clean] {msg}")


def check_inputs() -> None:
    if not DATASET_DIR.exists():
        sys.exit(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata.csv not found: {METADATA_IN}")
    if not WAVS_DIR.exists():
        sys.exit(f"[ERROR] wavs/ folder not found: {WAVS_DIR}")


def main() -> None:
    check_inputs()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output folder ready: {CLEAN_DIR}")

    # Read metadata — pipe-separated, no header: id | raw_text | normalized_text
    rows: list[dict] = []
    with METADATA_IN.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for line_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                log(f"  [WARN] Line {line_num} has fewer than 2 fields — skipping")
                continue
            original_id = row[0].strip()
            # Use normalized transcription (col 3) when available, fall back to raw (col 2)
            text = row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip()
            rows.append({"original_id": original_id, "text": text})
            if len(rows) == NUM_SAMPLES:
                break

    log(f"Read {len(rows)} samples from metadata.csv")

    out_rows: list[dict] = []
    skipped = 0

    for idx, entry in enumerate(rows, start=1):
        original_id = entry["original_id"]
        src_wav = WAVS_DIR / f"{original_id}.wav"

        if not src_wav.exists():
            log(f"  [WARN] Missing wav — skipping {original_id}: {src_wav}")
            skipped += 1
            continue

        clean_name = f"clean_{idx:04d}.wav"
        dst_wav = CLEAN_DIR / clean_name

        shutil.copy2(src_wav, dst_wav)
        log(f"  Copied {original_id}.wav → {clean_name}")

        out_rows.append(
            {
                "id": f"clean_{idx:04d}",
                "original_id": original_id,
                "clean_path": str(dst_wav.relative_to(ROOT)),
                "text": entry["text"],
            }
        )

    # Write output CSV
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "original_id", "clean_path", "text"])
        writer.writeheader()
        writer.writerows(out_rows)

    log(f"Wrote {len(out_rows)} entries to {METADATA_OUT}")
    if skipped:
        log(f"[WARN] Skipped {skipped} samples due to missing wav files")
    log("Phase 1 complete.")


if __name__ == "__main__":
    main()
