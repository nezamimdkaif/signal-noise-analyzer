#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_sine(f0=5.0, fs=500.0, duration=2.0, amplitude=1.0, phase=0.0):
    """Generate a clean sine wave."""
    t = np.arange(0, duration, 1.0/fs)
    x = amplitude * np.sin(2*np.pi*f0*t + phase)
    return t, x

def add_awgn(x, snr_db=None, noise_std=None, rng=None):
    """Add white Gaussian noise.
    - If snr_db is given, noise power is set for the desired SNR vs x.
    - Else if noise_std is given, use that directly.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if snr_db is not None:
        sig_power = np.mean(x**2)
        noise_power = sig_power / (10**(snr_db/10.0))
        noise_std = np.sqrt(noise_power)
    elif noise_std is None:
        noise_std = 0.2
    n = rng.normal(0.0, noise_std, size=x.shape)
    return x + n, n

def moving_average(x, window=11):
    """Simple moving average filter (odd window preferred).
    Pads at the boundaries using edge values.
    """
    if window < 1:
        return x.copy()
    kernel = np.ones(int(window))/float(window)
    # pad edges to avoid shift
    pad = (window-1)//2
    xpad = np.pad(x, (pad, pad), mode='edge')
    y = np.convolve(xpad, kernel, mode='valid')
    return y

def snr_db(reference, test, eps=1e-12):
    """Compute SNR in dB between clean reference and test signal."""
    num = np.sum(reference**2)
    den = np.sum((test - reference)**2) + eps
    return 10.0*np.log10(num/den)

def magnitude_spectrum(x, fs):
    """Return one-sided magnitude spectrum (freqs, mag)."""
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    mag = np.abs(X) / N
    return freqs, mag

def main():
    parser = argparse.ArgumentParser(description="Signal Noise Analyzer (NumPy + Matplotlib)")
    parser.add_argument("--f0", type=float, default=5.0, help="Sine frequency (Hz)")
    parser.add_argument("--fs", type=float, default=500.0, help="Sampling rate (Hz)")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration (s)")
    parser.add_argument("--snr_db", type=float, default=0.0, help="Target SNR of the noisy signal in dB")
    parser.add_argument("--noise_std", type=float, default=None, help="Alternative to snr_db: set noise std directly")
    parser.add_argument("--window", type=int, default=11, help="Moving average window size (odd number preferred)")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to ./outputs instead of just showing")
    args = parser.parse_args()

    # Generate clean signal
    t, clean = generate_sine(f0=args.f0, fs=args.fs, duration=args.duration)

    # Add noise
    noisy, noise = add_awgn(clean, snr_db=args.snr_db, noise_std=args.noise_std)

    # Filter
    filtered = moving_average(noisy, window=args.window)

    # Compute SNRs
    snr_noisy = snr_db(clean, noisy)
    snr_filt = snr_db(clean, filtered)

    print(f"SNR (Noisy)   : {snr_noisy:.2f} dB")
    print(f"SNR (Filtered): {snr_filt:.2f} dB")
    print(f"Improvement   : {snr_filt - snr_noisy:.2f} dB")

    # Prepare output directory if saving
    outdir = Path("outputs")
    if args.save_plots:
        outdir.mkdir(exist_ok=True)

    # Plot 1: Clean signal (time)
    plt.figure()
    plt.title("Clean Signal (Time)")
    plt.plot(t, clean)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if args.save_plots:
        plt.savefig(outdir / "01_clean_time.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Plot 2: Noisy signal (time)
    plt.figure()
    plt.title(f"Noisy Signal (Time) — SNR ~ {snr_noisy:.2f} dB")
    plt.plot(t, noisy)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if args.save_plots:
        plt.savefig(outdir / "02_noisy_time.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Plot 3: Filtered signal (time)
    plt.figure()
    plt.title(f"Filtered Signal (Time) — SNR ~ {snr_filt:.2f} dB")
    plt.plot(t, filtered)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if args.save_plots:
        plt.savefig(outdir / "03_filtered_time.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Plot 4: Spectrum — Noisy
    freqs_n, mag_n = magnitude_spectrum(noisy, fs=args.fs)
    plt.figure()
    plt.title("Magnitude Spectrum — Noisy")
    plt.plot(freqs_n, mag_n)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    if args.save_plots:
        plt.savefig(outdir / "04_noisy_spectrum.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # Plot 5: Spectrum — Filtered
    freqs_f, mag_f = magnitude_spectrum(filtered, fs=args.fs)
    plt.figure()
    plt.title("Magnitude Spectrum — Filtered")
    plt.plot(freqs_f, mag_f)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    if args.save_plots:
        plt.savefig(outdir / "05_filtered_spectrum.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
