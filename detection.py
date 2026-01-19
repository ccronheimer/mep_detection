#!/usr/bin/env python3
"""
Configurable Hemisphere/Phase-Aware TMS/MEP Detection Script (WINDOW + PEAK-BASED)

Key features:
1) PEAK-based pulse detection (find_peaks) instead of rising-edge threshold crossing.
2) Per-phase time windows (useful when early pulses didn’t record, e.g., Mely).
3) Backward compatible: if a config has hemisphere_switch_time (no "phases"), it behaves like
   the old script: healthy=[0, switch), stroke=[switch, end).
4) QA outputs:
   - ISI statistics per phase
   - Overlay plot (rectified signal with detected peaks) for first 30s of each phase

Usage:
  python detection_windowed.py "Dec1_Mely" --config Dec1_Mely --fs 2000
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENT_CONFIGS = {
"Dec1_Andy": {
    "description": "Dec 1 Andy Experiment",
    "hemisphere_switch_time": 1238.4,

    "healthy": {
        "channels": [135, 132, 130],
        "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
        "expected_pulses": 164,
        "min_isi_s": 3.5,
        "peak_prominence_mult": 8.0,
        "description": "Healthy: Left hemisphere → Right muscles (pulses 1–164)"
    },

    "stroke": {
        "channels": [136, 137, 139],
        "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
        "expected_pulses": 121,
        "min_isi_s": 6.0,
        "peak_prominence_mult": 8.0,
        "description": "Stroke: Right hemisphere → Left muscles (pulses 165–285)"
    }
},

    # --- Windowed config for Mely ---
    "Dec1_Mely": {
        "description": "Dec 1 Mely (windowed)",
        "fs_hint": 2000,
        "phases": {
            "healthy": {
                "time_window_s": [472.3, 697.0],
                "channels": [130, 132, 135],
                "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
                "expected_pulses": 120,
                "min_isi_s": 1.1,
                "peak_prominence_mult": 6.0,
                "description": "Healthy pulses 131–250 (Right muscles)"
            },
            "stroke": {
                "time_window_s": [697.0, None],
                "channels": [139, 137, 136],
                "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
                "expected_pulses": 122,
                "min_isi_s": 2.0,
                "peak_prominence_mult": 6.0,
                "description": "Stroke pulses 251–372 (Left muscles)"
            }
        }
    },

    # --- Existing configs (switch-time mode) ---
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "hemisphere_switch_time": 695.0,
        "healthy": {
            "channels": [2, 4, 7],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 182,
            "description": "Healthy Left Hemisphere → Right Muscles (0-695s)"
        },
        "stroke": {
            "channels": [8, 9, 11],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 263,
            "description": "Stroke Right Hemisphere → Left Muscles (695s-end)"
        }
    },

    "Nov5_Chive": {
        "description": "Nov5 Chive Experiment (Ch2,10 noisy)",
        "hemisphere_switch_time": 857.0,
        "healthy": {
            "channels": [9, 2, 8],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 115,
            "description": "Healthy Left Hemisphere → Right Muscles"
        },
        "stroke": {
            "channels": [4, 10, 7],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 172,
            "description": "Stroke Right Hemisphere → Left Muscles"
        }
    },

    "Nov5_Cheddar": {
        "description": "Nov5 Cheddar Experiment",
        "hemisphere_switch_time": 650.0,
        "healthy": {
            "channels": [2, 4, 12],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 105,
            "description": "Healthy Left Hemisphere → Right Muscles"
        },
        "stroke": {
            "channels": [14, 9, 13],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 195,
            "description": "Stroke Right Hemisphere → Left Muscles"
        }
    },

    "Oct31_Chive": {
        "description": "Oct31 Chive Experiment",
        "hemisphere_switch_time": 930.0,
        "healthy": {
            "channels": [1, 2, 3],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 85,
            "description": "Healthy Left Hemisphere → Right Muscles"
        },
        "stroke": {
            "channels": [4, 5, 6],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 205,
            "description": "Stroke Right Hemisphere → Left Muscles"
        }
    }
}

# ============================================================================
# HELPERS / QA
# ============================================================================

def get_experiment_config(folder_path, config_name=None):
    folder = Path(folder_path)
    folder_name = folder.name

    if config_name and config_name in EXPERIMENT_CONFIGS:
        config = EXPERIMENT_CONFIGS[config_name].copy()
        print(f"📋 Using specified config: {config_name}")
        return config

    if folder_name in EXPERIMENT_CONFIGS:
        config = EXPERIMENT_CONFIGS[folder_name].copy()
        print(f"📋 Found exact match config: {folder_name}")
        return config

    for key in EXPERIMENT_CONFIGS:
        if key.lower() in folder_name.lower() or folder_name.lower() in key.lower():
            config = EXPERIMENT_CONFIGS[key].copy()
            print(f"📋 Found partial match config: {key} for folder {folder_name}")
            return config

    default_key = list(EXPERIMENT_CONFIGS.keys())[0]
    config = EXPERIMENT_CONFIGS[default_key].copy()
    print(f"⚠️ No matching config found for '{folder_name}', using default: {default_key}")
    return config


def detect_available_channels(folder: Path):
    chan_files = list(folder.glob("chan*.mat"))
    available = []
    for f in chan_files:
        m = re.search(r"chan(\d+)\.mat", f.name)
        if m:
            available.append(int(m.group(1)))
    available.sort()
    print(f"🔍 Detected channels: {available}")
    return available


def qa_isi_stats(ts, detected_indices, label):
    if len(detected_indices) < 2:
        print(f"🧪 QA {label}: not enough pulses for ISI stats")
        return
    t = ts[np.array(detected_indices, dtype=int)]
    isi = np.diff(t)
    print(
        f"🧪 QA {label} ISI (s): median={np.median(isi):.3f}, mean={np.mean(isi):.3f}, "
        f"p10={np.percentile(isi,10):.3f}, p90={np.percentile(isi,90):.3f}, min={isi.min():.3f}"
    )


def qa_overlay_plot(ts, rect_signal, global_peaks, t0, t1, title):
    mask = (ts >= t0) & (ts <= t1)
    if not np.any(mask):
        return
    plt.figure(figsize=(12, 4))
    plt.plot(ts[mask], rect_signal[mask], lw=1)
    pk = np.array(global_peaks, dtype=int)
    pk = pk[(ts[pk] >= t0) & (ts[pk] <= t1)]
    if len(pk):
        plt.scatter(ts[pk], rect_signal[pk], s=25)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Rectified EMG")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def enforce_min_isi(peaks, min_samples):
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return peaks
    out = [int(peaks[0])]
    last = out[0]
    for p in peaks[1:]:
        if int(p) - last >= int(min_samples):
            out.append(int(p))
            last = int(p)
    return np.array(out, dtype=int)


# ============================================================================
# IO + FILTERING
# ============================================================================

def load_emg_data_flexible(folder: Path):
    ts_path = folder / "Timestamps.mat"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {ts_path}")

    ts = sio.loadmat(str(ts_path))["analogInputDataTime_s"].flatten()

    chan_files = list(folder.glob("chan*.mat"))
    chan_info = []
    for f in chan_files:
        m = re.search(r"chan(\d+)\.mat", f.name)
        if m:
            chan_info.append((int(m.group(1)), f))

    chan_info.sort(key=lambda x: x[0])
    chan_numbers = [x[0] for x in chan_info]

    n_chan = len(chan_info)
    emg = np.zeros((n_chan, len(ts)), dtype=np.float32)

    for i, (chan_num, file_path) in enumerate(chan_info):
        data = sio.loadmat(str(file_path))["chandata"].flatten()
        n = min(len(data), len(ts))
        emg[i, :n] = data[:n].astype(np.float32)
        if len(data) != len(ts):
            print(f"⚠️ Length mismatch chan{chan_num}: {len(data)} vs ts {len(ts)} (truncated to {n})")

    print(f"📊 Loaded channels: {chan_numbers}")
    return ts, emg, chan_numbers


def butterworth_filter_superior(emg, fs, lowcut=10, highcut=500, order=2):
    print(f"🔧 Applying {order}-order Butterworth filter: {lowcut}-{highcut} Hz")
    nyq = fs / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = sig.butter(order, [low, high], btype="band")
    filtered = sig.filtfilt(b, a, emg, axis=1)
    rectified = np.abs(filtered)
    print(f"✅ Filtered signal range: {filtered.min():.3f} to {filtered.max():.3f}")
    return filtered, rectified


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


# ============================================================================
# DETECTION CORE
# ============================================================================

def optimize_detection_threshold_peaks(signal, fs, expected_n, min_isi_s=0.6, peak_prom_mult=6.0):
    signal = np.asarray(signal, dtype=np.float64)

    distance = max(1, int(min_isi_s * fs))
    baseline = np.median(signal)
    scale = mad(signal)
    if scale <= 0:
        scale = float(np.std(signal) + 1e-9)

    best_peaks = np.array([], dtype=int)
    best_score = float("inf")

    for mult in np.arange(2.0, 25.0, 0.25):
        height = baseline + mult * scale
        prominence = peak_prom_mult * scale

        peaks, _props = sig.find_peaks(signal, height=height, prominence=prominence, distance=distance)
        peaks = enforce_min_isi(peaks, distance)

        count_err = abs(len(peaks) - expected_n)

        # Use peak heights as a stable "strength" bonus (since we enforced peaks after find_peaks)
        strength_bonus = -float(np.median(signal[peaks])) if len(peaks) else 0.0

        score = count_err * 1e6 + strength_bonus
        if score < best_score:
            best_score = score
            best_peaks = peaks

    return best_peaks.astype(int)


def calculate_snr(signal, peaks, fs):
    if len(peaks) == 0:
        return 0.0
    pre_samples = int(0.2 * fs)
    p2p = []
    for p in peaks:
        if p - pre_samples >= 0 and p + pre_samples < len(signal):
            w = signal[p - pre_samples:p + pre_samples + 1]
            p2p.append(float(np.max(w) - np.min(w)))
    if not p2p:
        return 0.0
    base_std = float(np.std(signal[:pre_samples])) if pre_samples < len(signal) else float(np.std(signal))
    return float(np.mean(p2p) / base_std) if base_std > 0 else 0.0


def normalize_config_to_phases(config: dict, ts: np.ndarray):
    if "phases" in config:
        return config["phases"]

    switch_time = float(config.get("hemisphere_switch_time", ts[-1]))
    return {
        "healthy": {**config["healthy"], "time_window_s": [0.0, switch_time]},
        "stroke": {**config["stroke"], "time_window_s": [switch_time, None]},
    }


def time_window_to_indices(ts: np.ndarray, start_s: float, end_s):
    if start_s is None:
        start_s = 0.0
    start_idx = int(np.searchsorted(ts, start_s, side="left"))

    if end_s is None:
        end_idx = len(ts)
    else:
        end_idx = int(np.searchsorted(ts, float(end_s), side="left"))

    end_idx = max(start_idx + 1, min(end_idx, len(ts)))
    return start_idx, end_idx


def detect_phase_peaks(emg_rect, chan_numbers, ts, fs, phase_name, phase_cfg, enable_overlay=True):
    target_channels = phase_cfg["channels"]
    expected_pulses = int(phase_cfg.get("expected_pulses", 0))
    min_isi_s = float(phase_cfg.get("min_isi_s", 0.6))
    peak_prom_mult = float(phase_cfg.get("peak_prominence_mult", 6.0))

    win = phase_cfg.get("time_window_s", [0.0, None])
    start_s, end_s = win[0], (win[1] if len(win) > 1 else None)

    start_idx, end_idx = time_window_to_indices(ts, start_s, end_s)
    seg_ts = ts[start_idx:end_idx]
    seg_emg = emg_rect[:, start_idx:end_idx]

    print(f"\n🧠 {phase_cfg.get('description', phase_name)}")
    print(f"   Target channels: {target_channels}")
    print(f"   Expected pulses: {expected_pulses}")
    print(f"   min_isi_s: {min_isi_s}, peak_prominence_mult: {peak_prom_mult}")
    print(f"   Window: {seg_ts[0]:.1f}s - {seg_ts[-1]:.1f}s (n={len(seg_ts)})")

    available_target = [ch for ch in target_channels if ch in chan_numbers]
    missing = [ch for ch in target_channels if ch not in chan_numbers]
    if missing:
        print(f"   ⚠️ Missing channels: {missing}")
    if not available_target:
        print(f"   ❌ No target channels available for {phase_name}")
        return None

    results = []
    for ch in available_target:
        idx = chan_numbers.index(ch)
        signal = seg_emg[idx]

        peaks = optimize_detection_threshold_peaks(
            signal=signal,
            fs=fs,
            expected_n=expected_pulses,
            min_isi_s=min_isi_s,
            peak_prom_mult=peak_prom_mult,
        )

        err = abs(len(peaks) - expected_pulses)
        snr = calculate_snr(signal, peaks, fs)
        results.append((err, -snr, ch, idx, peaks, snr))
        print(f"   Ch {ch}: {len(peaks)} pulses, error={err}, SNR={snr:.2f}")

    results.sort()
    _, _, best_ch, best_idx, best_peaks, _best_snr = results[0]

    global_peaks = best_peaks + start_idx

    qa_isi_stats(ts, global_peaks, phase_name)

    if len(global_peaks):
        print(
            f"   ✅ Selected Ch{best_ch} | detected={len(global_peaks)} | "
            f"first={ts[global_peaks[0]]:.2f}s last={ts[global_peaks[-1]]:.2f}s"
        )
    else:
        print(f"   ✅ Selected Ch{best_ch} | detected=0")

    if enable_overlay and len(global_peaks):
        best_rect_global = emg_rect[chan_numbers.index(best_ch)]
        t0 = float(seg_ts[0])
        t1 = float(min(seg_ts[0] + 30.0, seg_ts[-1]))
        qa_overlay_plot(ts, best_rect_global, global_peaks, t0, t1, f"{phase_name} QA overlay (Ch{best_ch})")

    return {
        "phase": phase_name,
        "channel_num": int(best_ch),
        "channel_idx": int(best_idx),
        "detected_indices": global_peaks.astype(int),
        "time_range": (float(seg_ts[0]), float(seg_ts[-1])),
        "config": phase_cfg,
    }


def run_detection(emg_rect, chan_numbers, ts, fs, config, enable_overlay=True):
    print("🎯 PHASE-AWARE PULSE DETECTION (WINDOW + PEAK)")
    print("=" * 55)

    phases = normalize_config_to_phases(config, ts)
    all_detections = []
    phase_results = {}

    for phase_name in ["healthy", "stroke"]:
        if phase_name not in phases:
            continue

        res = detect_phase_peaks(
            emg_rect, chan_numbers, ts, fs, phase_name, phases[phase_name], enable_overlay=enable_overlay
        )
        if res is None:
            continue

        phase_results[phase_name] = res

        for idx in res["detected_indices"]:
            all_detections.append(
                {
                    "pulse_index": int(idx),
                    "pulse_time_s": float(ts[idx]),
                    "phase": phase_name,
                    "channel_used": int(res["channel_num"]),
                }
            )

    all_detections.sort(key=lambda d: d["pulse_index"])
    return all_detections, phase_results


# ============================================================================
# EPOCHING + AMPLITUDE
# ============================================================================

def extract_meps(emg_filtered, all_detections, ts, fs):
    print("\n📡 EXTRACTING MEPs")
    print("=" * 30)

    pre_samples = int(50 * fs / 1000)
    post_samples = int(150 * fs / 1000)

    n_chan = emg_filtered.shape[0]
    epochs = []
    info = []

    for d in all_detections:
        pulse_idx = d["pulse_index"]
        if pulse_idx - pre_samples >= 0 and pulse_idx + post_samples < emg_filtered.shape[1]:
            epoch = emg_filtered[:, pulse_idx - pre_samples : pulse_idx + post_samples]
            epochs.append(epoch)
            info.append(
                {
                    "detection_idx": len(epochs) - 1,
                    "pulse_index": pulse_idx,
                    "pulse_time_s": float(ts[pulse_idx]),
                    "phase": d["phase"],
                    "detection_channel": d["channel_used"],
                }
            )

    if epochs:
        epochs = np.array(epochs)
    else:
        epochs = np.zeros((0, n_chan, pre_samples + post_samples), dtype=np.float32)

    time_vector = np.arange(-pre_samples, post_samples) / fs * 1000.0
    print(f"✅ Extracted {len(info)} epochs")
    return epochs, time_vector, info


def compute_amplitudes(epochs, time_vector, detection_info, chan_numbers, fs, config):
    print("\n📏 COMPUTING MEP AMPLITUDES")
    print("=" * 30)

    n_epochs, n_chan, n_samples = epochs.shape
    baseline_ms = 50
    mep_window_ms = [15, 50]

    stim_idx = int(np.argmin(np.abs(time_vector)))
    spm = fs / 1000.0
    mep_start = max(stim_idx + 1, stim_idx + int(mep_window_ms[0] * spm))
    mep_end = min(n_samples - 1, stim_idx + int(mep_window_ms[1] * spm))
    baseline_end = int(baseline_ms * spm)

    phases = normalize_config_to_phases(config, np.array([0.0, 1.0]))
    phase_channels = {k: set(v.get("channels", [])) for k, v in phases.items()}

    rows = []
    for eidx in range(min(n_epochs, len(detection_info))):
        meta = detection_info[eidx]
        phase = meta["phase"]
        relevant = phase_channels.get(phase, set())

        row = {
            "detection_idx": eidx,
            "pulse_index": meta["pulse_index"],
            "pulse_time_s": meta["pulse_time_s"],
            "phase": phase,
            "detection_channel": meta["detection_channel"],
        }

        for i, ch in enumerate(chan_numbers):
            if i >= n_chan:
                continue

            sig_ep = epochs[eidx, i, :]
            if 0 < baseline_end < len(sig_ep):
                sig_ep = sig_ep - np.mean(sig_ep[:baseline_end])

            mep = sig_ep[mep_start:mep_end] if mep_end <= len(sig_ep) else np.array([])
            amp = float(np.max(mep) - np.min(mep)) if len(mep) else np.nan

            row[f"ch{ch}_amplitude"] = amp
            row[f"ch{ch}_relevant"] = (ch in relevant)

        rows.append(row)

    print(f"✅ Computed amplitudes for {len(rows)} epochs")
    return rows


# ============================================================================
# PLOTTING
# ============================================================================

def create_summary_plots(df, config):
    print("\n📈 Creating summary plots...")
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"MEP Detection Results: {config.get('description','')}", fontsize=14, fontweight="bold")

    healthy = df[df["phase"] == "healthy"]
    stroke = df[df["phase"] == "stroke"]

    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(healthy["pulse_time_s"] / 60.0, np.ones(len(healthy)), s=10, alpha=0.7, label=f"healthy n={len(healthy)}")
    ax1.scatter(stroke["pulse_time_s"] / 60.0, np.zeros(len(stroke)), s=10, alpha=0.7, label=f"stroke n={len(stroke)}")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["stroke", "healthy"])
    ax1.set_xlabel("Time (min)")
    ax1.set_title("Detection timeline")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    phases = normalize_config_to_phases(config, np.array([0.0, 1.0]))

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Healthy hand amplitude dist")
    if len(healthy) > 0 and "healthy" in phases:
        hand_ch = phases["healthy"]["channels"][-1]
        col = f"ch{hand_ch}_amplitude"
        amps = healthy[col].dropna() if col in healthy.columns else pd.Series([], dtype=float)
        if len(amps):
            ax2.hist(amps, bins=15, alpha=0.8)
            ax2.set_xlabel("Amplitude")
            ax2.set_ylabel("Count")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Stroke hand amplitude dist")
    if len(stroke) > 0 and "stroke" in phases:
        hand_ch = phases["stroke"]["channels"][-1]
        col = f"ch{hand_ch}_amplitude"
        amps = stroke[col].dropna() if col in stroke.columns else pd.Series([], dtype=float)
        if len(amps):
            ax3.hist(amps, bins=15, alpha=0.8)
            ax3.set_xlabel("Amplitude")
            ax3.set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MEP Detection (window + peak based)")
    parser.add_argument("folder", type=Path, help="Session folder containing Timestamps.mat and chan*.mat")
    parser.add_argument("--fs", type=float, default=2000.0, help="Sampling rate (nf3=2000)")
    parser.add_argument("--config", type=str, help="Experiment config name")
    parser.add_argument("--no-qa-plots", action="store_true", help="Disable QA overlay plots")
    args = parser.parse_args()

    print("=" * 70)
    print("    SIMPLE CONFIGURABLE MEP DETECTION (WINDOW + PEAK BASED)")
    print("=" * 70)

    config = get_experiment_config(args.folder, args.config)

    detect_available_channels(args.folder)
    ts, emg_raw, chan_numbers = load_emg_data_flexible(args.folder)

    raw_range = float(np.max(np.abs(emg_raw)))
    if raw_range < 1.0:
        emg_scaled = emg_raw * 1e3
        print("📊 Signal scaling: µV (scaled)")
    else:
        emg_scaled = emg_raw
        print("📊 Signal scaling: µV")

    emg_filtered, emg_rect = butterworth_filter_superior(emg_scaled, args.fs)

    all_detections, phase_results = run_detection(
        emg_rect, chan_numbers, ts, args.fs, config, enable_overlay=(not args.no_qa_plots)
    )

    epochs, time_vector, detection_info = extract_meps(emg_filtered, all_detections, ts, args.fs)

    amplitude_rows = compute_amplitudes(epochs, time_vector, detection_info, chan_numbers, args.fs, config)
    df = pd.DataFrame(amplitude_rows)

    create_summary_plots(df, config)

    output_file = args.folder / "mep_results.csv"
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("           DETECTION COMPLETE")
    print("=" * 70)

    healthy_n = int((df["phase"] == "healthy").sum()) if len(df) else 0
    stroke_n = int((df["phase"] == "stroke").sum()) if len(df) else 0
    print(f"Experiment: {config.get('description','')}")
    print("Mode: windowed phases" if "phases" in config else "Mode: switch-time (backward compatible)")
    print(f"Healthy detected: {healthy_n}")
    print(f"Stroke detected:  {stroke_n}")
    print(f"Total epochs:     {len(df)}")
    print(f"💾 Saved: {output_file}")

    if "phases" in config:
        for pname, pres in phase_results.items():
            if pres and len(pres["detected_indices"]):
                first_t = ts[pres["detected_indices"][0]]
                last_t = ts[pres["detected_indices"][-1]]
                print(f"🔎 {pname}: first={first_t:.2f}s last={last_t:.2f}s ch={pres['channel_num']} n={len(pres['detected_indices'])}")


if __name__ == "__main__":
    main()
