"""
Phase 6.5 — Validate extracted Wav2Vec2 features and clean Mel-spectrogram targets.
Checks shape, dimension, NaN/Inf, and time-step alignment.
"""

import csv
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_features.csv"
RESULTS_DIR = ROOT / "results"
REPORT_CSV = RESULTS_DIR / "feature_check.csv"
SUMMARY_TXT = RESULTS_DIR / "feature_summary.txt"

# Expected dimensions
EXPECTED_W2V_DIM = 768
EXPECTED_MEL_BINS = 80


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[check_features] {msg}")


def has_nan_inf(arr: np.ndarray) -> tuple[bool, bool]:
    return bool(np.any(np.isnan(arr))), bool(np.any(np.isinf(arr)))


def check_npz(npz_path: Path) -> dict:
    """
    Load and validate one .npz file.
    Returns a result dict; 'valid' is True only when all checks pass.
    """
    result = {
        "file_exists": False,
        "has_w2v": False,
        "has_mel": False,
        "w2v_shape": None,
        "mel_shape": None,
        "w2v_time_steps": None,
        "mel_time_steps": None,
        "timestep_diff": None,
        "w2v_dim_ok": False,
        "mel_bins_ok": False,
        "w2v_nan": False,
        "w2v_inf": False,
        "mel_nan": False,
        "mel_inf": False,
        "valid": False,
        "error": "",
    }

    if not npz_path.exists():
        result["error"] = "file not found"
        return result
    result["file_exists"] = True

    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as exc:
        result["error"] = f"load error: {exc}"
        return result

    # ── Key presence ───────────────────────────────────────────────────────────
    result["has_w2v"] = "wav2vec2_features" in data
    result["has_mel"] = "clean_mel" in data

    if not result["has_w2v"]:
        result["error"] = "missing key: wav2vec2_features"
        return result
    if not result["has_mel"]:
        result["error"] = "missing key: clean_mel"
        return result

    w2v: np.ndarray = data["wav2vec2_features"]
    mel: np.ndarray = data["clean_mel"]

    # ── Shapes ─────────────────────────────────────────────────────────────────
    result["w2v_shape"] = "x".join(str(s) for s in w2v.shape)
    result["mel_shape"] = "x".join(str(s) for s in mel.shape)

    if w2v.ndim == 2:
        result["w2v_time_steps"] = w2v.shape[0]
        result["w2v_dim_ok"] = w2v.shape[1] == EXPECTED_W2V_DIM
    else:
        result["error"] = f"wav2vec2_features has unexpected ndim={w2v.ndim} (expected 2)"
        return result

    if mel.ndim == 2:
        result["mel_time_steps"] = mel.shape[1]
        result["mel_bins_ok"] = mel.shape[0] == EXPECTED_MEL_BINS
    else:
        result["error"] = f"clean_mel has unexpected ndim={mel.ndim} (expected 2)"
        return result

    result["timestep_diff"] = result["w2v_time_steps"] - result["mel_time_steps"]

    # ── NaN / Inf ──────────────────────────────────────────────────────────────
    result["w2v_nan"], result["w2v_inf"] = has_nan_inf(w2v)
    result["mel_nan"], result["mel_inf"] = has_nan_inf(mel)

    # ── Overall validity ───────────────────────────────────────────────────────
    checks_passed = (
        result["w2v_dim_ok"]
        and result["mel_bins_ok"]
        and not result["w2v_nan"]
        and not result["w2v_inf"]
        and not result["mel_nan"]
        and not result["mel_inf"]
    )
    result["valid"] = checks_passed
    if not checks_passed and not result["error"]:
        failures = []
        if not result["w2v_dim_ok"]:
            failures.append(f"w2v dim≠{EXPECTED_W2V_DIM}")
        if not result["mel_bins_ok"]:
            failures.append(f"mel bins≠{EXPECTED_MEL_BINS}")
        if result["w2v_nan"]:
            failures.append("w2v has NaN")
        if result["w2v_inf"]:
            failures.append("w2v has Inf")
        if result["mel_nan"]:
            failures.append("mel has NaN")
        if result["mel_inf"]:
            failures.append("mel has Inf")
        result["error"] = "; ".join(failures)

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_features.csv not found: {METADATA_IN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log(f"Loaded {len(rows)} entries from metadata_features.csv")
    log("")

    report_rows: list[dict] = []
    w2v_steps_list: list[int] = []
    mel_steps_list: list[int] = []
    valid_count = 0

    for entry in rows:
        row_id = entry["id"]
        npz_path = ROOT / entry["feature_path"]
        result = check_npz(npz_path)

        status = "OK" if result["valid"] else "FAIL"
        w2v_ts = result["w2v_time_steps"]
        mel_ts = result["mel_time_steps"]
        diff = result["timestep_diff"]

        log(
            f"  [{status}] {row_id}"
            f"  w2v={result['w2v_shape']}"
            f"  mel={result['mel_shape']}"
            + (f"  Δt={diff:+d}" if diff is not None else "")
            + (f"  ✗ {result['error']}" if result["error"] else "")
        )

        if result["valid"]:
            valid_count += 1
            w2v_steps_list.append(w2v_ts)
            mel_steps_list.append(mel_ts)

        report_rows.append({
            "id": row_id,
            "feature_path": entry["feature_path"],
            "status": status,
            "w2v_shape": result["w2v_shape"],
            "mel_shape": result["mel_shape"],
            "w2v_time_steps": w2v_ts,
            "mel_time_steps": mel_ts,
            "timestep_diff": diff,
            "w2v_dim_ok": result["w2v_dim_ok"],
            "mel_bins_ok": result["mel_bins_ok"],
            "w2v_nan": result["w2v_nan"],
            "w2v_inf": result["w2v_inf"],
            "mel_nan": result["mel_nan"],
            "mel_inf": result["mel_inf"],
            "error": result["error"],
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    total = len(rows)
    invalid_count = total - valid_count
    avg_w2v = round(sum(w2v_steps_list) / len(w2v_steps_list), 1) if w2v_steps_list else 0.0
    avg_mel = round(sum(mel_steps_list) / len(mel_steps_list), 1) if mel_steps_list else 0.0
    diffs = [w - m for w, m in zip(w2v_steps_list, mel_steps_list)]
    avg_diff = round(sum(diffs) / len(diffs), 1) if diffs else 0.0
    train_ready = valid_count == total and total > 0

    log("")
    log("=" * 60)
    log(f"  Total feature files       : {total}")
    log(f"  Valid files               : {valid_count}")
    log(f"  Invalid files             : {invalid_count}")
    log(f"  Avg wav2vec2 time steps   : {avg_w2v}")
    log(f"  Avg mel time steps        : {avg_mel}")
    log(f"  Avg time-step difference  : {avg_diff:+.1f}  (w2v − mel)")
    log(f"  Train-ready?              : {'YES' if train_ready else 'NO — fix issues above'}")
    log("=" * 60)

    # ── Write report CSV ───────────────────────────────────────────────────────
    fieldnames = [
        "id", "feature_path", "status",
        "w2v_shape", "mel_shape",
        "w2v_time_steps", "mel_time_steps", "timestep_diff",
        "w2v_dim_ok", "mel_bins_ok",
        "w2v_nan", "w2v_inf", "mel_nan", "mel_inf",
        "error",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    log(f"Detailed report → {REPORT_CSV}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    note_diff = (
        "Δt is expected: distorted audio is ~7% longer (speed ×1.07) so\n"
        "  wav2vec2 frames > mel frames. Alignment must be handled in Phase 7."
    )
    summary_lines = [
        "Speech Re-Synthesis POC — Feature Check Summary",
        "=" * 52,
        f"Total feature files       : {total}",
        f"Valid files               : {valid_count}",
        f"Invalid files             : {invalid_count}",
        f"Avg wav2vec2 time steps   : {avg_w2v}",
        f"Avg mel time steps        : {avg_mel}",
        f"Avg time-step difference  : {avg_diff:+.1f}  (w2v − mel)",
        f"Train-ready?              : {'YES' if train_ready else 'NO'}",
        "",
        f"Note: {note_diff}",
        "",
        f"Report CSV                : {REPORT_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary        → {SUMMARY_TXT}")
    log("Phase 6.5 complete.")


if __name__ == "__main__":
    main()
