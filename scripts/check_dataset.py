"""
Phase 3 — Validate and inspect the dataset.
Reads metadata_poc.csv, checks every audio file, and writes reports to results/.
"""

import csv
import sys
from pathlib import Path

import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_poc.csv"
RESULTS_DIR = ROOT / "results"
REPORT_CSV = RESULTS_DIR / "dataset_check.csv"
SUMMARY_TXT = RESULTS_DIR / "dataset_summary.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[check_dataset] {msg}")


def check_inputs() -> None:
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_poc.csv not found: {METADATA_IN}")


def inspect_audio(path: Path) -> dict:
    """Return audio properties or an error dict if the file cannot be read."""
    if not path.exists():
        return {"exists": False, "sample_rate": None, "num_samples": None,
                "duration_s": None, "channels": None, "error": "file not found"}
    try:
        info = sf.info(str(path))
        return {
            "exists": True,
            "sample_rate": info.samplerate,
            "num_samples": info.frames,
            "duration_s": round(info.duration, 4),
            "channels": info.channels,
            "error": "",
        }
    except Exception as exc:
        return {"exists": False, "sample_rate": None, "num_samples": None,
                "duration_s": None, "channels": None, "error": str(exc)}


def channel_label(n: int | None) -> str:
    if n is None:
        return "unknown"
    return {1: "mono", 2: "stereo"}.get(n, f"{n}-ch")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    check_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Results folder ready: {RESULTS_DIR}")

    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Loaded {len(rows)} rows from metadata_poc.csv")
    log("")

    report_rows: list[dict] = []
    clean_durations: list[float] = []
    dist_durations: list[float] = []
    missing_count = 0

    for entry in rows:
        row_id = entry["id"]
        clean_path = ROOT / entry["clean_path"]
        dist_path = ROOT / entry["distorted_path"]

        clean_info = inspect_audio(clean_path)
        dist_info = inspect_audio(dist_path)

        clean_ok = clean_info["exists"]
        dist_ok = dist_info["exists"]
        row_missing = (not clean_ok) + (not dist_ok)
        missing_count += row_missing

        status = "OK" if (clean_ok and dist_ok) else "MISSING"

        log(
            f"  [{status}] {row_id}"
            f" | clean: {clean_info['duration_s']}s"
            f" {channel_label(clean_info['channels'])}"
            f" {clean_info['sample_rate']}Hz"
            f" ({clean_info['num_samples']} samples)"
            f" | distorted: {dist_info['duration_s']}s"
            f" {channel_label(dist_info['channels'])}"
            f" {dist_info['sample_rate']}Hz"
            f" ({dist_info['num_samples']} samples)"
        )

        if clean_ok and clean_info["duration_s"] is not None:
            clean_durations.append(clean_info["duration_s"])
        if dist_ok and dist_info["duration_s"] is not None:
            dist_durations.append(dist_info["duration_s"])

        report_rows.append({
            "id": row_id,
            "original_id": entry["original_id"],
            "text_length": len(entry["text"]),
            # clean
            "clean_exists": clean_ok,
            "clean_sr": clean_info["sample_rate"],
            "clean_samples": clean_info["num_samples"],
            "clean_duration_s": clean_info["duration_s"],
            "clean_channels": channel_label(clean_info["channels"]),
            "clean_error": clean_info["error"],
            # distorted
            "dist_exists": dist_ok,
            "dist_sr": dist_info["sample_rate"],
            "dist_samples": dist_info["num_samples"],
            "dist_duration_s": dist_info["duration_s"],
            "dist_channels": channel_label(dist_info["channels"]),
            "dist_error": dist_info["error"],
            "status": status,
        })

    # ── Statistics ─────────────────────────────────────────────────────────────
    total = len(rows)
    valid_clean = len(clean_durations)
    valid_dist = len(dist_durations)
    avg_clean = round(sum(clean_durations) / valid_clean, 4) if valid_clean else 0.0
    avg_dist = round(sum(dist_durations) / valid_dist, 4) if valid_dist else 0.0
    total_clean_s = round(sum(clean_durations), 2)
    total_dist_s = round(sum(dist_durations), 2)

    log("")
    log("=" * 60)
    log(f"  Total rows            : {total}")
    log(f"  Valid clean files     : {valid_clean} / {total}")
    log(f"  Valid distorted files : {valid_dist} / {total}")
    log(f"  Missing files         : {missing_count}")
    log(f"  Avg clean duration    : {avg_clean} s")
    log(f"  Avg distorted duration: {avg_dist} s")
    log(f"  Total clean audio     : {total_clean_s} s ({total_clean_s/60:.2f} min)")
    log(f"  Total distorted audio : {total_dist_s} s ({total_dist_s/60:.2f} min)")
    log("=" * 60)

    # ── Write report CSV ───────────────────────────────────────────────────────
    fieldnames = [
        "id", "original_id", "text_length",
        "clean_exists", "clean_sr", "clean_samples", "clean_duration_s", "clean_channels", "clean_error",
        "dist_exists", "dist_sr", "dist_samples", "dist_duration_s", "dist_channels", "dist_error",
        "status",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    log(f"Detailed report → {REPORT_CSV}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    summary_lines = [
        "Speech Re-Synthesis POC — Dataset Check Summary",
        "=" * 50,
        f"Total rows            : {total}",
        f"Valid clean files     : {valid_clean} / {total}",
        f"Valid distorted files : {valid_dist} / {total}",
        f"Missing files         : {missing_count}",
        f"Avg clean duration    : {avg_clean} s",
        f"Avg distorted duration: {avg_dist} s",
        f"Total clean audio     : {total_clean_s} s ({total_clean_s/60:.2f} min)",
        f"Total distorted audio : {total_dist_s} s ({total_dist_s/60:.2f} min)",
        "",
        f"Report CSV            : {REPORT_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary text  → {SUMMARY_TXT}")
    log("Phase 3 complete.")


if __name__ == "__main__":
    main()
