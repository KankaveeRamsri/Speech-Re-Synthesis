"""
Phase 4.6 — STT evaluation across all distortion levels: clean, mild, medium, severe.
Verifies whether stronger distortion causes higher WER.
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
METADATA_IN = ROOT / "data" / "metadata_multilevel_poc.csv"
RESULTS_DIR = ROOT / "results"
OUT_CSV = RESULTS_DIR / "multilevel_stt_results.csv"
OUT_SUMMARY = RESULTS_DIR / "multilevel_stt_summary.txt"

DEFAULT_N = 10
DEFAULT_MODEL = "base"

LEVELS = ["clean", "mild", "medium", "severe"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[multilevel_stt_eval] {msg}")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return min(jiwer.wer(ref, hyp), 1.0)


def transcribe(model: whisper.Whisper, path: Path) -> str:
    if not path.exists():
        log(f"    [WARN] File not found — returning empty: {path.name}")
        return ""
    try:
        result = model.transcribe(str(path), language="en", fp16=False)
        return result["text"].strip()
    except Exception as exc:
        log(f"    [WARN] Transcription failed for {path.name}: {exc}")
        return ""


def path_col(level: str) -> str:
    """Map level name to the matching column in metadata_multilevel_poc.csv."""
    return "clean_path" if level == "clean" else f"distorted_{level}_path"


def trunc(text: str, n: int = 75) -> str:
    return text[:n] + ("..." if len(text) > n else "")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.6 — multi-level STT evaluation")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Samples to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_multilevel_poc.csv not found: {METADATA_IN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_IN.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[: args.n]
    log(f"Evaluating {len(rows)} / {len(all_rows)} samples across {len(LEVELS)} levels")

    log(f"Loading Whisper model: '{args.model}' ...")
    model = whisper.load_model(args.model)
    log("Model ready.")
    log("")

    # ── Per-sample loop ────────────────────────────────────────────────────────
    out_rows: list[dict] = []
    wer_accum: dict[str, list[float]] = {lvl: [] for lvl in LEVELS}

    for i, entry in enumerate(rows, start=1):
        row_id = entry["id"]
        reference = entry["text"]

        log(f"[{i}/{len(rows)}] {row_id}")
        log(f"  ref    : {trunc(reference)}")

        preds: dict[str, str] = {}
        wers: dict[str, float] = {}

        for level in LEVELS:
            audio_path = ROOT / entry[path_col(level)]
            pred = transcribe(model, audio_path)
            w = safe_wer(reference, pred)
            preds[level] = pred
            wers[level] = w
            wer_accum[level].append(w)
            log(f"  {level:7s}: {trunc(pred)}  →  WER {w:.4f}")

        log("")

        out_rows.append({
            "id": row_id,
            "reference": reference,
            "clean_prediction": preds["clean"],
            "mild_prediction": preds["mild"],
            "medium_prediction": preds["medium"],
            "severe_prediction": preds["severe"],
            "clean_wer": round(wers["clean"], 4),
            "mild_wer": round(wers["mild"], 4),
            "medium_wer": round(wers["medium"], 4),
            "severe_wer": round(wers["severe"], 4),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(out_rows)
    avg: dict[str, float] = {
        lvl: round(sum(wer_accum[lvl]) / n, 4) if n else 0.0
        for lvl in LEVELS
    }

    # Check monotonic increase clean → mild → medium → severe
    ordered = [avg["clean"], avg["mild"], avg["medium"], avg["severe"]]
    wer_increases = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))

    log("=" * 60)
    log(f"  Samples evaluated  : {n}")
    for lvl in LEVELS:
        log(f"  Avg {lvl:7s} WER  : {avg[lvl]:.4f}  ({avg[lvl]*100:.1f}%)")
    log(f"  WER increases with distortion? : {'YES' if wer_increases else 'NOT STRICTLY'}")
    log("=" * 60)

    # ── Write results CSV ──────────────────────────────────────────────────────
    fieldnames = [
        "id", "reference",
        "clean_prediction", "mild_prediction", "medium_prediction", "severe_prediction",
        "clean_wer", "mild_wer", "medium_wer", "severe_wer",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log(f"Results CSV  → {OUT_CSV}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    # Build a simple bar chart (each █ ≈ 2 WER %)
    def bar(wer_val: float) -> str:
        blocks = round(wer_val * 50)   # 50 blocks = 100 %
        return "█" * blocks or "▏"

    summary_lines = [
        "Speech Re-Synthesis POC — Multi-Level STT Evaluation Summary",
        "=" * 58,
        f"Whisper model              : {args.model}",
        f"Samples evaluated          : {n}",
        "",
        "Average WER per level:",
        f"  clean   {avg['clean']:.4f} ({avg['clean']*100:5.1f}%)  {bar(avg['clean'])}",
        f"  mild    {avg['mild']:.4f} ({avg['mild']*100:5.1f}%)  {bar(avg['mild'])}",
        f"  medium  {avg['medium']:.4f} ({avg['medium']*100:5.1f}%)  {bar(avg['medium'])}",
        f"  severe  {avg['severe']:.4f} ({avg['severe']*100:5.1f}%)  {bar(avg['severe'])}",
        "",
        f"WER increases with distortion? : {'YES' if wer_increases else 'NOT STRICTLY'}",
        "",
        f"Results CSV : {OUT_CSV}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary TXT  → {OUT_SUMMARY}")
    log("Phase 4.6 complete.")


if __name__ == "__main__":
    main()
