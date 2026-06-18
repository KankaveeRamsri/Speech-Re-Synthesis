# Speech Re-Synthesis POC

A proof-of-concept pipeline for evaluating speech degradation and restoration using the LJ Speech dataset. The project simulates distorted audio at multiple severity levels, runs ASR evaluation using OpenAI Whisper, and applies a simple signal-processing enhancement baseline — all without any model training.

---

## Goal

Demonstrate a full speech re-synthesis evaluation loop:

1. Prepare clean speech subset
2. Simulate degraded (distorted) speech at multiple levels
3. Measure intelligibility loss using Word Error Rate (WER)
4. Apply a signal-processing enhancement baseline
5. Measure whether enhancement recovers intelligibility

---

## Phases Completed

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `prepare_clean.py` | Extract 50-sample subset from LJ Speech |
| 2 | `create_distortion.py` | Single-level distortion (medium baseline) |
| 3 | `check_dataset.py` | Validate all audio files and report stats |
| 4 | `run_stt_eval.py` | Baseline STT WER: clean vs distorted |
| 4.5 | `create_multilevel_distortion.py` | Generate mild / medium / severe distortions |
| 4.6 | `run_multilevel_stt_eval.py` | WER across all 4 levels |
| 5 | `simple_enhancement.py` | Signal-processing enhancement (no training) |
| 5.5 | `run_enhanced_stt_eval.py` | WER: distorted vs enhanced |
| 6 | `extract_wav2vec2_features.py` | Extract Wav2Vec2 hidden states + clean Mel targets |
| 6.5 | `check_features.py` | Validate .npz features (shape, NaN/Inf, alignment) |

---

## Key Results (10-sample evaluation, Whisper `base`)

| Track | Avg WER |
|-------|---------|
| Clean | 7.0% |
| Mild distorted | 6.3% |
| Medium distorted | 10.2% |
| Severe distorted | ~88–92% |
| Enhanced medium | ~10.1% |
| Enhanced severe | ~88–89% |

Signal-processing enhancement alone has minimal impact — confirming that model-based enhancement (Wav2Vec2, HiFi-GAN) is needed for meaningful recovery, especially on severe distortion.

---

## Folder Structure

```
Speech-Re-Synthesis/
├── scripts/                        # All runnable Python scripts
│   ├── prepare_clean.py            # Phase 1
│   ├── create_distortion.py        # Phase 2
│   ├── check_dataset.py            # Phase 3
│   ├── run_stt_eval.py             # Phase 4
│   ├── create_multilevel_distortion.py  # Phase 4.5
│   ├── run_multilevel_stt_eval.py  # Phase 4.6
│   ├── simple_enhancement.py       # Phase 5
│   ├── run_enhanced_stt_eval.py    # Phase 5.5
│   ├── extract_wav2vec2_features.py  # Phase 6
│   └── check_features.py           # Phase 6.5
├── data/                           # Dataset and generated audio (mostly gitignored)
│   └── LJSpeech-1.1 3/            # ← NOT included; download separately (see below)
├── results/                        # CSV and TXT evaluation reports (committed)
│   ├── dataset_check.csv
│   ├── dataset_summary.txt
│   ├── stt_results.csv
│   ├── stt_summary.txt
│   ├── multilevel_stt_results.csv
│   ├── multilevel_stt_summary.txt
│   ├── enhanced_stt_results.csv
│   ├── enhanced_stt_summary.txt
│   ├── feature_check.csv
│   └── feature_summary.txt
├── requirements.txt
└── README.md
```

> Audio files (`data/clean/`, `data/distorted_*/`, `data/enhanced_*/`) and generated CSVs are excluded from Git. Run the scripts in order to regenerate them.

---

## Dataset

This project uses [LJ Speech 1.1](https://keithito.com/LJ-Speech-Dataset/) — a public domain English speech dataset.

**The dataset is NOT included in this repository.**

Download and place it at:

```
data/LJSpeech-1.1 3/
├── metadata.csv
├── README
└── wavs/
    ├── LJ001-0001.wav
    └── ...
```

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Whisper model weights (~140 MB for `base`) are downloaded automatically on first run.

---

## Running the Pipeline

Run scripts in order from the project root:

```bash
# Phase 1 — prepare 50-sample clean subset
python scripts/prepare_clean.py

# Phase 2 — create single-level distorted audio
python scripts/create_distortion.py

# Phase 3 — validate dataset
python scripts/check_dataset.py

# Phase 4 — baseline STT evaluation (10 samples, Whisper base)
python scripts/run_stt_eval.py
python scripts/run_stt_eval.py --n 50 --model small   # full run

# Phase 4.5 — create mild / medium / severe distortions
python scripts/create_multilevel_distortion.py

# Phase 4.6 — multi-level STT evaluation
python scripts/run_multilevel_stt_eval.py
python scripts/run_multilevel_stt_eval.py --n 50

# Phase 5 — simple signal-processing enhancement
python scripts/simple_enhancement.py

# Phase 5.5 — evaluate enhanced audio
python scripts/run_enhanced_stt_eval.py
python scripts/run_enhanced_stt_eval.py --n 50

# Phase 6 — extract Wav2Vec2 features + clean Mel targets (10 samples default)
python scripts/extract_wav2vec2_features.py
python scripts/extract_wav2vec2_features.py --n 50   # all samples

# Phase 6.5 — validate extracted features
python scripts/check_features.py
```

---

## Enhancement Pipeline (Phase 5)

Three signal-processing stages applied to distorted audio — no model training:

1. **Spectral noise reduction** — `noisereduce` (non-stationary mode, prop_decrease=0.80)
2. **Band-pass filter** — Butterworth order 5, pass-band 80 Hz – 7500 Hz
3. **Peak normalisation** — scale to 0.95 FS

---

## Distortion Parameters

| Level | Volume | Noise σ | Low-pass | Speed |
|-------|--------|---------|----------|-------|
| mild | 70% | 0.003 | 5000 Hz | ×1.03 |
| medium | 45% | 0.008 | 3400 Hz | ×1.07 |
| severe | 20% | 0.030 | 2000 Hz | ×1.15 |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading and resampling |
| `soundfile` | WAV file I/O |
| `scipy` | Butterworth filters |
| `numpy` | Array operations |
| `openai-whisper` | ASR transcription |
| `jiwer` | Word Error Rate calculation |
| `noisereduce` | Spectral noise reduction |
| `torch` | Tensor computation for Wav2Vec2 inference |
| `transformers` | Wav2Vec2Model + Processor (Hugging Face) |
| `pandas` | Data manipulation |

---

## Feature Format (Phase 6)

Each `.npz` file in `data/features_wav2vec2/` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `wav2vec2_features` | `(T_w2v, 768)` | Hidden states from distorted_medium audio |
| `clean_mel` | `(80, T_mel)` | Log-Mel spectrogram from clean audio |
| `id` | scalar | Sample ID string |
| `text` | scalar | Normalised transcript |
| `clean_path` | scalar | Relative path to clean wav |
| `distorted_medium_path` | scalar | Relative path to distorted wav |

> `T_w2v > T_mel` by ~22 frames on average because distorted audio is 7% longer (speed ×1.07). Temporal alignment is a Phase 7 concern.

---

## Next Steps (not yet implemented)

- Phase 7: Train a Wav2Vec2 → Mel-spectrogram mapping network
- Phase 8: HiFi-GAN vocoder to convert predicted Mel back to waveform
- Objective metrics: PESQ, STOI, SI-SNR
