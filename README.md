# Concurrent Translation Pipeline on Edge Device

Real-time speech translation system running entirely on NVIDIA Jetson Nano (4GB)

**English speech → ASR (Whisper) → MT (Opus-MT) → QE (mBERT) → TTS (espeak-ng)**

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-369/)
[![JetPack 4.6](https://img.shields.io/badge/JetPack-4.6-green.svg)](https://developer.nvidia.com/embedded/jetpack)

---

## Overview

Privacy-preserving edge translation pipeline designed for resource-constrained devices. All processing happens on-device without cloud dependencies.

### Key Features

- **On-Device Processing:** No internet required, full privacy
- **Integrated Quality Estimation:** Real-time translation quality scoring using mBERT
- **Noise Reduction:** Adaptive filtering for noisy environments (scipy + noisereduce)
- **Modular Architecture:** Easy to swap ASR/MT/QE/TTS components
- **Memory Optimized:** Runs on 4GB RAM with sequential model loading
- **Voice Activity Detection:** Automatic speech detection with silence-based segmentation

### Performance

- End-to-end latency: ~5-8s for 10s audio (target: real-time capable)
- Memory footprint: ~3.2GB peak usage
- Hardware: NVIDIA Jetson Nano 4GB, 128-core Maxwell GPU

---

## Hardware Requirements

- **Device:** NVIDIA Jetson Nano 4GB (or higher)
- **OS:** Ubuntu 18.04 (JetPack 4.6)
- **Python:** 3.6.9
- **Storage:** ~5GB free space (models + code)
- **Microphone:** USB or 3.5mm input

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/juebae/concurrent_translation.git
cd concurrent_translation
```

### 2. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng portaudio19-dev libsndfile1
```

### 3. Install Python Packages

```bash
pip3 install -r requirements.txt
```

### 4. Install Vosk ASR

Vosk requires manual installation on Jetson (aarch64):

```bash
# Download Vosk wheel
wget https://github.com/alphacep/vosk-api/releases/download/v0.3.42/vosk-0.3.42-py3-none-linux_aarch64.whl

# Install
pip3 install vosk-0.3.42-py3-none-linux_aarch64.whl

# Download ASR model (~200 MB)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
rm vosk-model-en-us-0.22-lgraph.zip
```

### 5. Test Installation

```bash
# Test Vosk ASR standalone
python3 test_vosk_simple.py

# Test full pipeline (1 recording)
python3 phase3b_main_FIXED.py --num-recordings 1
```

---

## Usage

### Interactive Mode (Continuous Translation)

```bash
python3 phase3b_main_FIXED.py
```

Press Enter to start recording, speak naturally, system auto-detects silence.

### Batch Mode (Fixed Number of Recordings)

```bash
python3 phase3b_main_FIXED.py --num-recordings 10
```

### Change Target Language

```bash
python3 phase3b_main_FIXED.py --target-lang fr  # Spanish by default
```

### List Audio Devices

```bash
python3 phase3b_main_FIXED.py --list-devices
```

---

## Architecture

```
┌─────────────┐
│  Microphone │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Noise Reduction    │  (scipy + noisereduce)
│  Voice Activity Det │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   ASR (Vosk)        │  English transcription
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  MT (Opus-MT)       │  Translation to target lang
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  QE (mBERT)         │  Quality scoring (0-1)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  TTS (espeak-ng)    │  Audio synthesis & playback
└─────────────────────┘
```

---

## Project Structure

```
concurrent_translation/
├── phase3a_asr_module.py      # ASR component (Vosk)
├── phase3a_mt_module.py       # Machine Translation (Opus-MT)
├── phase3a_qe_module.py       # Quality Estimation (mBERT)
├── phase3a_tts_module.py      # Text-to-Speech (espeak-ng)
├── phase3b_audio_module.py    # Microphone capture + VAD + noise reduction
├── phase3b_main_FIXED.py      # Main pipeline orchestrator
├── test_vosk.py               # Vosk standalone test
├── test_vosk_simple.py        # Simple Vosk test
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
└── README.md                  # This file
```

---

## Technical Details

### Component Specifications

| Component | Model/Library | Size | Device | Purpose |
|-----------|---------------|------|--------|---------|
| **ASR** | Vosk small-en-us-0.22 | 200 MB | CPU | Speech recognition |
| **MT** | Opus-MT-en-es | ~300 MB | GPU | Neural translation |
| **QE** | mBERT base | ~400 MB | GPU | Translation quality |
| **TTS** | espeak-ng | ~5 MB | CPU | Speech synthesis |

### Why Vosk Instead of Whisper?

**Problem:** OpenAI Whisper requires Python ≥3.8  
**Constraint:** JetPack 4.6 is limited to Python 3.6  
**Solution:** Vosk provides competitive ASR accuracy with Python 3.6 support

### Noise Reduction Strategy

1. **Calibration:** 2-second ambient noise profile capture at startup
2. **Filtering:** noisereduce library with scipy signal processing
3. **Result:** Improved ASR accuracy in real-world noisy environments

### Memory Management

- **Sequential Loading:** Models loaded one at a time to fit in 4GB
- **GPU Sharing:** MT and QE share GPU, ASR uses CPU
- **Peak Usage:** ~3.2GB during translation phase

---

## Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| ASR Latency | ~3-4s | For 10s audio |
| MT Latency | ~0.8-1.2s | Per sentence |
| QE Latency | ~1.0-1.5s | Per translation |
| TTS Latency | ~1.5-2.0s | Per sentence |
| **Total Pipeline** | ~5-8s | End-to-end |
| Real-time Factor | 0.5-0.8 | Faster than real-time |

*(Baseline measurements, Week 3)*

---

## Testing

### Unit Tests

```bash
# Test ASR only
python3 test_vosk_simple.py

# Test each module
python3 -c "from phase3a_asr_module import WhisperASR; print('ASR OK')"
python3 -c "from phase3a_mt_module import OpusMT; print('MT OK')"
```

### Integration Test

```bash
# Single end-to-end test
python3 phase3b_main_FIXED.py --num-recordings 1
```

---

## Dissertation Context

This project is part of a Masters dissertation investigating:

> **"Integrated Quality Estimation in Real-Time Edge Speech Translation"**

### Research Questions

1. Can concurrent execution reduce latency vs serial pipeline?
2. Can integrated QE improve quality without major overhead?
3. Is real-time translation feasible on 4GB edge devices?

### Current Status

- **Week 3:** Baseline implementation complete (serial pipeline)
- **Week 4:** Baseline testing & metrics collection (upcoming)
- **Week 5-6:** Concurrent execution implementation
- **Week 7-8:** Analysis & write-up

---

## Known Issues & Limitations

- **Python 3.6 constraint:** Limits library choices (JetPack 4.6 requirement)
- **ASR accuracy:** Small Vosk model trades accuracy for memory
- **GPU memory:** MT + QE must run sequentially (not enough VRAM for parallel)
- **Language support:** Currently only English → Spanish (easily extensible)

---

## Troubleshooting

### "No module named 'vosk'"

Install Vosk wheel manually (see installation section).

### "Model not found: vosk-model-en-us-0.22-lgraph"

Download model:

```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
```

### "CUDA out of memory"

Reduce batch size or use CPU fallback for MT/QE.

### Poor ASR quality

- Check microphone input: `python3 phase3b_main_FIXED.py --list-devices`
- Increase noise reduction sensitivity in `phase3b_audio_module.py`
- Use larger Vosk model (trades memory for accuracy)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourlastname2026edge,
  title={Integrated Quality Estimation in Real-Time Edge Speech Translation},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

---

## License

Academic research project. Contact for usage permissions.

---

## Acknowledgments

- **Vosk:** Lightweight ASR by Alpha Cephei
- **Opus-MT:** Neural MT by Helsinki-NLP
- **Transformers:** Hugging Face library
- **NVIDIA:** JetPack SDK & Jetson hardware

---

## Contact

- **Author:** Zubair
- **Institution:** [Your University]
- **GitHub:** [@juebae](https://github.com/juebae)

---

**Last Updated:** February 16, 2026
