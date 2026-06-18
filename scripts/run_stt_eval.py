"""
Phase 4 — Baseline STT evaluation: clean vs distorted audio.
Uses OpenAI Whisper (base model) + jiwer for WER calculation.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import jiwer
import whisper

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
METADATA_IN = ROOT / "data" / "metadata_poc.csv"
RESULTS_DIR = ROOT / "results"
STT_CSV = RESULTS_DIR / "stt_results.csv"
STT_SUMMARY = RESULTS_DIR / "stt_summary.txt"

DEFAULT_N_SAMPLES = 10
DEFAULT_MODEL = "base"


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[run_stt_eval] {msg}")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_wer(reference: str, hypothesis: str) -> float:
    """Return WER capped at 1.0; handle empty strings gracefully."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return min(jiwer.wer(ref, hyp), 1.0)


def transcribe(model: whisper.Whisper, path: Path) -> str:
    """Transcribe a wav file; return empty string on any error."""
    if not path.exists():
        log(f"    [WARN] File not found — skipping: {path}")
        return ""
    try:
        result = model.transcribe(str(path), language="en", fp16=False)
        return result["text"].strip()
    except Exception as exc:
        log(f"    [WARN] Transcription failed for {path.name}: {exc}")
        return ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — STT evaluation")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Number of samples to evaluate (default: {DEFAULT_N_SAMPLES})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_poc.csv not found: {METADATA_IN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ──────────────────────────────────────────────────────────
    with METADATA_IN.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[: args.n]
    log(f"Evaluating {len(rows)} / {len(all_rows)} samples")

    # ── Load Whisper model ─────────────────────────────────────────────────────
    log(f"Loading Whisper model: '{args.model}' ...")
    model = whisper.load_model(args.model)
    log("Model ready.")
    log("")

    # ── Transcribe & score ─────────────────────────────────────────────────────
    out_rows: list[dict] = []
    clean_wers: list[float] = []
    dist_wers: list[float] = []

    for i, entry in enumerate(rows, start=1):
        row_id = entry["id"]
        reference = entry["text"]
        clean_path = ROOT / entry["clean_path"]
        dist_path = ROOT / entry["distorted_path"]

        log(f"[{i}/{len(rows)}] {row_id}")
        log(f"  ref : {reference[:80]}{'...' if len(reference) > 80 else ''}")

        clean_pred = transcribe(model, clean_path)
        dist_pred = transcribe(model, dist_path)

        log(f"  clean pred   : {clean_pred[:80]}{'...' if len(clean_pred) > 80 else ''}")
        log(f"  distort pred : {dist_pred[:80]}{'...' if len(dist_pred) > 80 else ''}")

        c_wer = safe_wer(reference, clean_pred)
        d_wer = safe_wer(reference, dist_pred)
        clean_wers.append(c_wer)
        dist_wers.append(d_wer)

        log(f"  WER → clean: {c_wer:.4f}  distorted: {d_wer:.4f}")
        log("")

        out_rows.append({
            "id": row_id,
            "reference": reference,
            "clean_path": entry["clean_path"],
            "distorted_path": entry["distorted_path"],
            "clean_prediction": clean_pred,
            "distorted_prediction": dist_pred,
            "clean_wer": round(c_wer, 4),
            "distorted_wer": round(d_wer, 4),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(out_rows)
    avg_clean_wer = round(sum(clean_wers) / n, 4) if n else 0.0
    avg_dist_wer = round(sum(dist_wers) / n, 4) if n else 0.0
    distorted_is_worse = avg_dist_wer > avg_clean_wer

    log("=" * 60)
    log(f"  Samples evaluated       : {n}")
    log(f"  Avg clean WER           : {avg_clean_wer:.4f}  ({avg_clean_wer*100:.1f}%)")
    log(f"  Avg distorted WER       : {avg_dist_wer:.4f}  ({avg_dist_wer*100:.1f}%)")
    log(f"  Distorted harder?       : {'YES' if distorted_is_worse else 'NO'}")
    log("=" * 60)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "id", "reference",
        "clean_path", "distorted_path",
        "clean_prediction", "distorted_prediction",
        "clean_wer", "distorted_wer",
    ]
    with STT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log(f"Results CSV  → {STT_CSV}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    summary_lines = [
        "Speech Re-Synthesis POC — STT Evaluation Summary",
        "=" * 50,
        f"Whisper model           : {args.model}",
        f"Samples evaluated       : {n}",
        f"Avg clean WER           : {avg_clean_wer:.4f}  ({avg_clean_wer*100:.1f}%)",
        f"Avg distorted WER       : {avg_dist_wer:.4f}  ({avg_dist_wer*100:.1f}%)",
        f"Distorted harder?       : {'YES' if distorted_is_worse else 'NO'}",
        "",
        f"Results CSV             : {STT_CSV}",
    ]
    STT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary TXT  → {STT_SUMMARY}")
    log("Phase 4 complete.")


if __name__ == "__main__":
    main()
