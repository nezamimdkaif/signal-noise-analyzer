# Signal Noise Analyzer 📡  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/github/license/nezamimdkaif/signal-noise-analyzer)](./LICENSE)  
[![Last Commit](https://img.shields.io/github/last-commit/nezamimdkaif/signal-noise-analyzer?color=green)](https://github.com/nezamimdkaif/signal-noise-analyzer/commits/main)  
[![Repo Size](https://img.shields.io/github/repo-size/nezamimdkaif/signal-noise-analyzer)](https://github.com/nezamimdkaif/signal-noise-analyzer)  
[![Stars](https://img.shields.io/github/stars/nezamimdkaif/signal-noise-analyzer?style=social)](https://github.com/nezamimdkaif/signal-noise-analyzer/stargazers)  

A small, ECE-focused project that generates a clean sine signal, adds noise, applies a simple low-pass filter (moving average), and computes **Signal-to-Noise Ratio (SNR)** before and after filtering. It also visualizes the clean, noisy, and filtered signals.

## Features
- Generate sine waves with configurable frequency, duration, and sample rate.
- Add AWGN (additive white Gaussian noise) for a target SNR (in dB) *or* a custom noise standard deviation.
- Apply a **moving average low-pass filter** (NumPy-only, no SciPy).
- Compute SNR (in dB) between the clean reference and any processed signal.
- Visualize:
  - Time-domain signals (clean / noisy / filtered)
  - Magnitude spectrum via FFT for noisy and filtered signals

## Why this project?
- It’s **branch-specific (ECE)** and great for interviews/portfolios.
- Shows understanding of **DSP basics** (noise, filtering, SNR).
- Uses only **NumPy + Matplotlib**, so it’s easy to run anywhere.

## Quick Start
```bash
# (Optional) create venv and install
pip install -r requirements.txt

# Run with defaults
python signal_noise_analyzer.py

# Customize parameters
python signal_noise_analyzer.py --f0 5 --fs 500 --duration 2.0 --snr_db 0 --window 15 --save_plots
```
Output images (PNG) are saved to `outputs/` when `--save_plots` is used.

## SNR Definition
We compute SNR (dB) between a **reference (clean)** signal `x` and a **test** signal `y` as:
\[ \text{SNR} = 10 \log_{10} \frac{\sum x^2}{\sum (y - x)^2 + \epsilon} \]
`\epsilon` avoids division by zero.

## Roadmap / Extensions
- Replace moving average with FIR windowed-sinc or IIR (Butterworth) filter.
- Add **Savitzky–Golay** smoothing for better shape preservation.
- Add GUI sliders (streamlit) for interactive exploration.
- Support multiple wave types (square/triangle) and multi-tone signals.

---

## Dataset Preview

| Image | Description |
|-------|-------------|
| ![Dataset1](dataset/Dataset1.png) | Sample image 1 – example content |
| ![Dataset2](dataset/Dataset2.png) | Sample image 2 – example content |
| ![Dataset3](dataset/Dataset3.png) | Sample image 3 – example content |
| ![Dataset4](dataset/Dataset4.png) | Sample image 4 – example content |
