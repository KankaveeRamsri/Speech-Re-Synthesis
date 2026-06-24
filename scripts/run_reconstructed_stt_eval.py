"""
Phase 10 — STT evaluation of Griffin-Lim reconstructed audio.

Transcribes data/reconstructed_audio/*.wav with Whisper and computes WER
against ground-truth text.  Also loads previous-phase WERs from
results/enhanced_stt_results.csv so all three tracks can be compared
side-by-side in the summary without re-running Whisper on them.

Outputs:
  results/reconstructed_stt_results.csv
  results/reconstructed_stt_summary.txt
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import jiwer
import whisper

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
METADATA_IN  = ROOT / "data"    / "metadata_reconstructed_audio.csv"
ENHANCED_CSV = ROOT / "results" / "enhanced_stt_results.csv"
RESULTS_DIR  = ROOT / "results"
OUT_CSV      = RESULTS_DIR / "reconstructed_stt_results.csv"
OUT_SUMMARY  = RESULTS_DIR / "reconstructed_stt_summary.txt"

DEFAULT_N     = 10
DEFAULT_MODEL = "base"

# WER threshold below which we consider the audio "usable" for STT
USABLE_WER_THRESHOLD = 0.30


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[reconstructed_stt_eval] {msg}")


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
        log(f"  [WARN] Audio file not found — returning empty: {path.name}")
        return ""
    try:
        result = model.transcribe(str(path), language="en", fp16=False)
        return result["text"].strip()
    except Exception as exc:
        log(f"  [WARN] Transcription failed for {path.name}: {exc}")
        return ""


def trunc(text: str, n: int = 68) -> str:
    return text[:n] + ("..." if len(text) > n else "")


def bar(wer_val: float, width: int = 40) -> str:
    """ASCII bar scaled to `width` characters = 100 % WER."""
    blocks = round(wer_val * width)
    return "█" * blocks or "▏"


# ── Load previous WERs for comparison ─────────────────────────────────────────

def load_enhanced_wers(csv_path: Path) -> dict[str, dict[str, float]]:
    """
    Returns {id: {"distorted_medium_wer": float, "enhanced_medium_wer": float}}
    from results/enhanced_stt_results.csv.  Returns {} if file is missing.
    """
    if not csv_path.exists():
        log(f"[INFO] Previous results not found, skipping comparison: {csv_path.name}")
        return {}
    out: dict[str, dict[str, float]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["id"]] = {
                    "distorted_medium_wer": float(row["distorted_medium_wer"]),
                    "enhanced_medium_wer":  float(row["enhanced_medium_wer"]),
                }
            except (KeyError, ValueError):
                pass
    log(f"Loaded previous WERs for {len(out)} samples from {csv_path.name}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 — reconstructed audio STT evaluation"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Samples to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # ── Validate ───────────────────────────────────────────────────────────────
    if not METADATA_IN.exists():
        sys.exit(f"[ERROR] Metadata not found: {METADATA_IN}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read metadata ──────────────────────────────────────────────────────────
    with METADATA_IN.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows[: args.n]
    log(f"Metadata: {len(all_rows)} total, evaluating {len(rows)}")

    # ── Load comparison WERs ───────────────────────────────────────────────────
    prev_wers = load_enhanced_wers(ENHANCED_CSV)

    # ── Load Whisper ───────────────────────────────────────────────────────────
    log(f"Loading Whisper model: '{args.model}' ...")
    model = whisper.load_model(args.model)
    log("Model ready.")
    log("")

    out_rows:   list[dict]    = []
    recon_wers: list[float]   = []

    for idx, entry in enumerate(rows, start=1):
        sample_id  = entry["id"]
        audio_path = ROOT / entry["reconstructed_audio_path"]
        reference  = entry.get("text", "")

        log(f"[{idx:>2}/{len(rows)}] {sample_id}")
        log(f"  ref        : {trunc(reference)}")

        # ── Transcribe ─────────────────────────────────────────────────────────
        pred  = transcribe(model, audio_path)
        r_wer = safe_wer(reference, pred)
        recon_wers.append(r_wer)

        log(f"  recon pred : {trunc(pred)}")
        log(f"  recon WER  : {r_wer:.4f}  ({r_wer*100:.1f}%)")

        # ── Previous WERs for this sample (if available) ───────────────────────
        prev = prev_wers.get(sample_id, {})
        if prev:
            dm_wer = prev["distorted_medium_wer"]
            em_wer = prev["enhanced_medium_wer"]
            delta  = r_wer - dm_wer
            log(f"  dist_med   : {dm_wer:.4f}  enh_med : {em_wer:.4f}  "
                f"Δrecon vs dist_med : {delta:+.4f}")
        log("")

        out_rows.append({
            "id":                      sample_id,
            "reference":               reference,
            "reconstructed_audio_path": entry["reconstructed_audio_path"],
            "reconstructed_prediction": pred,
            "reconstructed_wer":       round(r_wer, 4),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n          = len(out_rows)
    avg_recon  = round(sum(recon_wers) / n, 4) if n else 0.0

    # Comparison averages (only samples that appear in prev_wers)
    dm_vals = [prev_wers[r["id"]]["distorted_medium_wer"]
               for r in out_rows if r["id"] in prev_wers]
    em_vals = [prev_wers[r["id"]]["enhanced_medium_wer"]
               for r in out_rows if r["id"] in prev_wers]
    avg_dm  = round(sum(dm_vals) / len(dm_vals), 4) if dm_vals else None
    avg_em  = round(sum(em_vals) / len(em_vals), 4) if em_vals else None

    usable  = avg_recon <= USABLE_WER_THRESHOLD

    log("=" * 60)
    log(f"  Samples evaluated    : {n}")
    log(f"  Avg reconstructed WER: {avg_recon:.4f}  ({avg_recon*100:.1f}%)")
    if avg_dm is not None:
        log(f"  Avg distorted_medium : {avg_dm:.4f}  ({avg_dm*100:.1f}%)  [Phase 5.5]")
        log(f"  Avg enhanced_medium  : {avg_em:.4f}  ({avg_em*100:.1f}%)  [Phase 5.5]")
        log(f"  Reconstructed vs dist_med: {avg_recon - avg_dm:+.4f}")
    log(f"  Usable for STT?      : {'YES' if usable else 'NOT YET'}"
        f"  (threshold ≤ {USABLE_WER_THRESHOLD*100:.0f}%)")
    log("=" * 60)

    # ── Write results CSV ──────────────────────────────────────────────────────
    fieldnames = [
        "id", "reference", "reconstructed_audio_path",
        "reconstructed_prediction", "reconstructed_wer",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    log(f"Results CSV  → {OUT_CSV.relative_to(ROOT)}")

    # ── Write summary TXT ──────────────────────────────────────────────────────
    lines = [
        "Speech Re-Synthesis POC — Reconstructed Audio STT Evaluation (Phase 10)",
        "=" * 70,
        f"Whisper model         : {args.model}",
        f"Samples evaluated     : {n}",
        "",
        "── WER Results ────────────────────────────────────────────────────────",
        f"  reconstructed  WER  : {avg_recon:.4f} ({avg_recon*100:5.1f}%)  {bar(avg_recon)}",
    ]

    if avg_dm is not None:
        dm_delta = avg_recon - avg_dm
        em_delta = avg_recon - avg_em
        lines += [
            f"  distorted_medium WER: {avg_dm:.4f} ({avg_dm*100:5.1f}%)  {bar(avg_dm)}  [Phase 5.5]",
            f"  enhanced_medium  WER: {avg_em:.4f} ({avg_em*100:5.1f}%)  {bar(avg_em)}  [Phase 5.5]",
            "",
            "── Comparison vs Phase 5.5 ────────────────────────────────────────────",
            f"  Δ recon vs distorted_medium : {dm_delta:+.4f} ({dm_delta*100:+.1f}%)",
            f"  Δ recon vs enhanced_medium  : {em_delta:+.4f} ({em_delta*100:+.1f}%)",
        ]
    else:
        lines += [
            "",
            "  (Previous phase WERs not available — run Phase 5.5 for comparison)",
        ]

    lines += [
        "",
        "── Usability Assessment ───────────────────────────────────────────────",
        f"  Threshold             : ≤ {USABLE_WER_THRESHOLD*100:.0f}% WER to be considered usable",
        f"  Reconstructed audio   : {'USABLE for STT' if usable else 'NOT YET usable for STT'}",
    ]

    if not usable:
        lines += [
            "",
            "  Note: Griffin-Lim introduces phase artifacts that degrade Whisper",
            "  accuracy.  Expected next step: replace with HiFi-GAN vocoder",
            "  (Phase 11) and re-evaluate.",
        ]

    lines += [
        "",
        f"Results CSV  : {OUT_CSV.relative_to(ROOT)}",
    ]

    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Summary TXT  → {OUT_SUMMARY.relative_to(ROOT)}")
    log("Phase 10 complete.")


if __name__ == "__main__":
    main()
