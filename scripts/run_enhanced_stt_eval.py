"""
Phase 5.5 — STT evaluation: distorted vs enhanced audio (medium and severe).
Measures whether simple signal-processing enhancement improves Whisper WER.
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
METADATA_IN = ROOT / "data" / "metadata_enhanced_poc.csv"
RESULTS_DIR = ROOT / "results"
OUT_CSV = RESULTS_DIR / "enhanced_stt_results.csv"
OUT_SUMMARY = RESULTS_DIR / "enhanced_stt_summary.txt"

DEFAULT_N = 10
DEFAULT_MODEL = "base"

# Each tuple: (column_key, display_label)
TRACKS = [
    ("distorted_medium", "distorted_medium"),
    ("enhanced_medium",  "enhanced_medium"),
    ("distorted_severe", "distorted_severe"),
    ("enhanced_severe",  "enhanced_severe"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[enhanced_stt_eval] {msg}")


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


def trunc(text: str, n: int = 72) -> str:
    return text[:n] + ("..." if len(text) > n else "")


def col_name(key: str) -> str:
    """metadata_enhanced_poc.csv column name for a given track key."""
    return f"{key}_path"


def bar(wer_val: float) -> str:
    """ASCII bar: each █ ≈ 2 % WER."""
    blocks = round(wer_val * 50)
    return "█" * blocks or "▏"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.5 — enhanced STT evaluation")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Samples to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] metadata_enhanced_poc.csv not found: {METADATA_IN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_IN.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[: args.n]
    log(f"Evaluating {len(rows)} / {len(all_rows)} samples — 4 tracks per sample")

    log(f"Loading Whisper model: '{args.model}' ...")
    model = whisper.load_model(args.model)
    log("Model ready.")
    log("")

    out_rows: list[dict] = []
    wer_accum: dict[str, list[float]] = {key: [] for key, _ in TRACKS}

    for i, entry in enumerate(rows, start=1):
        row_id = entry["id"]
        reference = entry["text"]

        log(f"[{i}/{len(rows)}] {row_id}")
        log(f"  ref               : {trunc(reference)}")

        preds: dict[str, str] = {}
        wers: dict[str, float] = {}

        for key, label in TRACKS:
            path = ROOT / entry[col_name(key)]
            pred = transcribe(model, path)
            w = safe_wer(reference, pred)
            preds[key] = pred
            wers[key] = w
            wer_accum[key].append(w)
            log(f"  {label:24s}: {trunc(pred, 48)}  WER {w:.4f}")

        # inline improvement indicators
        med_delta = wers["enhanced_medium"] - wers["distorted_medium"]
        sev_delta = wers["enhanced_severe"] - wers["distorted_severe"]
        log(f"  medium improvement: {'YES ▼ ' + f'{abs(med_delta):.4f}' if med_delta < 0 else 'no  ▲ ' + f'{med_delta:.4f}'}")
        log(f"  severe improvement: {'YES ▼ ' + f'{abs(sev_delta):.4f}' if sev_delta < 0 else 'no  ▲ ' + f'{sev_delta:.4f}'}")
        log("")

        out_rows.append({
            "id": row_id,
            "reference": reference,
            "distorted_medium_prediction": preds["distorted_medium"],
            "enhanced_medium_prediction": preds["enhanced_medium"],
            "distorted_severe_prediction": preds["distorted_severe"],
            "enhanced_severe_prediction": preds["enhanced_severe"],
            "distorted_medium_wer": round(wers["distorted_medium"], 4),
            "enhanced_medium_wer": round(wers["enhanced_medium"], 4),
            "distorted_severe_wer": round(wers["distorted_severe"], 4),
            "enhanced_severe_wer": round(wers["enhanced_severe"], 4),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(out_rows)
    avg: dict[str, float] = {
        key: round(sum(wer_accum[key]) / n, 4) if n else 0.0
        for key, _ in TRACKS
    }

    med_improved = avg["enhanced_medium"] < avg["distorted_medium"]
    sev_improved = avg["enhanced_severe"] < avg["distorted_severe"]
    med_gain = avg["distorted_medium"] - avg["enhanced_medium"]
    sev_gain = avg["distorted_severe"] - avg["enhanced_severe"]

    log("=" * 60)
    log(f"  Samples evaluated        : {n}")
    log(f"  Avg distorted_medium WER : {avg['distorted_medium']:.4f}  ({avg['distorted_medium']*100:.1f}%)")
    log(f"  Avg enhanced_medium  WER : {avg['enhanced_medium']:.4f}  ({avg['enhanced_medium']*100:.1f}%)")
    log(f"  Medium improved?         : {'YES  (WER -' + f'{med_gain:.4f})' if med_improved else 'NO   (WER +' + f'{abs(med_gain):.4f})'}")
    log(f"  Avg distorted_severe WER : {avg['distorted_severe']:.4f}  ({avg['distorted_severe']*100:.1f}%)")
    log(f"  Avg enhanced_severe  WER : {avg['enhanced_severe']:.4f}  ({avg['enhanced_severe']*100:.1f}%)")
    log(f"  Severe improved?         : {'YES  (WER -' + f'{sev_gain:.4f})' if sev_improved else 'NO   (WER +' + f'{abs(sev_gain):.4f})'}")
    log("=" * 60)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "id", "reference",
        "distorted_medium_prediction", "enhanced_medium_prediction",
        "distorted_severe_prediction", "enhanced_severe_prediction",
        "distorted_medium_wer", "enhanced_medium_wer",
        "distorted_severe_wer", "enhanced_severe_wer",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log(f"Results CSV  → {OUT_CSV}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    summary_lines = [
        "Speech Re-Synthesis POC — Enhanced STT Evaluation Summary",
        "=" * 58,
        f"Whisper model              : {args.model}",
        f"Samples evaluated          : {n}",
        "",
        "Medium level:",
        f"  distorted_medium WER : {avg['distorted_medium']:.4f} ({avg['distorted_medium']*100:5.1f}%)  {bar(avg['distorted_medium'])}",
        f"  enhanced_medium  WER : {avg['enhanced_medium']:.4f} ({avg['enhanced_medium']*100:5.1f}%)  {bar(avg['enhanced_medium'])}",
        f"  Enhancement improved medium? : {'YES  (ΔWER -' + f'{med_gain*100:.1f}' + '%)' if med_improved else 'NO   (ΔWER +' + f'{abs(med_gain)*100:.1f}' + '%)'}",
        "",
        "Severe level:",
        f"  distorted_severe WER : {avg['distorted_severe']:.4f} ({avg['distorted_severe']*100:5.1f}%)  {bar(avg['distorted_severe'])}",
        f"  enhanced_severe  WER : {avg['enhanced_severe']:.4f} ({avg['enhanced_severe']*100:5.1f}%)  {bar(avg['enhanced_severe'])}",
        f"  Enhancement improved severe? : {'YES  (ΔWER -' + f'{sev_gain*100:.1f}' + '%)' if sev_improved else 'NO   (ΔWER +' + f'{abs(sev_gain)*100:.1f}' + '%)'}",
        "",
        f"Results CSV : {OUT_CSV}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log(f"Summary TXT  → {OUT_SUMMARY}")
    log("Phase 5.5 complete.")


if __name__ == "__main__":
    main()
