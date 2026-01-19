#!/usr/bin/env python3
"""
MEP Amplitude & Latency Detection Script
Enhanced version of your epoch viewer that calculates BOTH amplitude and latency
Integrates with your existing analysis workflow
"""
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

# Use your existing experiment configurations
EXPERIMENT_CONFIGS = {
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "hemisphere_switch_time": 695.0,
        "healthy": {
            "channels": [2, 4, 7],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 182,
        },
        "stroke": {
            "channels": [8, 9, 11],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 263,
        }
    },
    "Nov5_Chive": {
        "description": "Nov5 Chive Experiment",
        "hemisphere_switch_time": 857.0,
        "healthy": {
            "channels": [9, 2, 8],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 115,
        },
        "stroke": {
            "channels": [4, 10, 7],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 172,
        }
    },
    "Nov5_Cheddar": {
        "description": "Nov5 Cheddar Experiment",
        "hemisphere_switch_time": 650.0,
        "healthy": {
            "channels": [2, 4, 12],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 105,
        },
        "stroke": {
            "channels": [14, 9, 13],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 195,
        }
    },
    "Oct31_Chive": {
        "description": "Oct31 Chive Experiment",
        "hemisphere_switch_time": 930.0,
        "healthy": {
            "channels": [1, 2, 3],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 85,
        },
        "stroke": {
            "channels": [4, 5, 6],
            "channel_names": ["Bicep", "Brachioradialis", "APB"],
            "expected_pulses": 205,
        }
    },
}

# =============================================================================
# YOUR EXISTING FUNCTIONS (from epoch viewer)
# =============================================================================

def get_experiment_config(folder_path, config_name=None):
    """Your existing config function"""
    folder = Path(folder_path)
    folder_name = folder.name
    
    if config_name and config_name in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[config_name].copy()
    
    if folder_name in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[folder_name].copy()
    
    for key in EXPERIMENT_CONFIGS:
        if key.lower() in folder_name.lower():
            return EXPERIMENT_CONFIGS[key].copy()
    
    return EXPERIMENT_CONFIGS[list(EXPERIMENT_CONFIGS.keys())[0]].copy()

def load_emg_data_flexible(folder: Path):
    """Your existing data loading function"""
    ts_path = folder / "Timestamps.mat"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {ts_path}")
    
    ts = sio.loadmat(str(ts_path))['analogInputDataTime_s'].flatten()
    
    chan_files = list(folder.glob("chan*.mat"))
    chan_info = []
    
    for f in chan_files:
        import re
        match = re.search(r'chan(\d+)\.mat', f.name)
        if match:
            chan_num = int(match.group(1))
            chan_info.append((chan_num, f))
    
    chan_info.sort(key=lambda x: x[0])
    chan_numbers = [x[0] for x in chan_info]
    
    n_chan = len(chan_info)
    emg = np.zeros((n_chan, len(ts)), dtype=np.float32)
    
    for i, (chan_num, file_path) in enumerate(chan_info):
        data = sio.loadmat(str(file_path))['chandata'].flatten()
        emg[i] = data.astype(np.float32)
    
    # Auto-scale data
    raw_range = np.max(np.abs(emg))
    if raw_range < 1:
        emg = emg * 1e6  # V to µV
    elif raw_range < 1000:
        emg = emg * 1e3  # mV to µV
    
    print(f"📊 Loaded channels: {chan_numbers}")
    return ts, emg, chan_numbers

def find_stimulus_artifacts(signal, fs, min_amplitude_factor=5, min_isi_ms=800):
    """Your existing stimulus detection function"""
    median_val = np.median(signal)
    mad = np.median(np.abs(signal - median_val))
    threshold = median_val + min_amplitude_factor * mad
    
    pos_crossings = np.where(np.diff(signal > threshold) == 1)[0]
    neg_crossings = np.where(np.diff(signal < -threshold) == 1)[0]
    all_crossings = np.sort(np.concatenate([pos_crossings, neg_crossings]))
    
    if len(all_crossings) == 0:
        return np.array([])
    
    min_isi_samples = int(min_isi_ms * fs / 1000)
    filtered_crossings = [all_crossings[0]]
    
    for crossing in all_crossings[1:]:
        if crossing - filtered_crossings[-1] > min_isi_samples:
            filtered_crossings.append(crossing)
    
    return np.array(filtered_crossings)

# =============================================================================
# NEW: MEP LATENCY CALCULATION FUNCTIONS
# =============================================================================

def calculate_mep_latency_threshold(epoch, time_vector, baseline_window=(-50, -10),
                                   search_window=(5, 50), threshold_factor=2.5):
    """
    Calculate MEP latency using threshold method
    
    Returns:
        latency (ms), or None if no MEP detected
    """
    # Baseline correction
    baseline_idx = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    baseline_mean = np.mean(epoch[baseline_idx])
    baseline_std = np.std(epoch[baseline_idx])
    
    corrected = epoch - baseline_mean
    
    # Set threshold (2.5 SDs above baseline noise)
    threshold = threshold_factor * baseline_std
    
    # Search for first crossing in MEP window
    search_idx = (time_vector >= search_window[0]) & (time_vector <= search_window[1])
    search_signal = np.abs(corrected[search_idx])
    search_times = time_vector[search_idx]
    
    # Find first point above threshold
    above_threshold = np.where(search_signal > threshold)[0]
    
    if len(above_threshold) > 0:
        # Require at least 3 consecutive points above threshold (reduces noise)
        for i in range(len(above_threshold) - 2):
            if (above_threshold[i+1] == above_threshold[i] + 1 and 
                above_threshold[i+2] == above_threshold[i] + 2):
                return search_times[above_threshold[i]]
        # If no 3 consecutive, return first crossing
        return search_times[above_threshold[0]]
    
    return None

def calculate_mep_features(epoch, time_vector, baseline_window=(-50, -10),
                          mep_window=(15, 50), latency_method='threshold'):
    """
    Calculate BOTH amplitude AND latency for a single epoch
    
    Returns:
        dict with amplitude, latency, peak_time, baseline_std, valid
    """
    # 1. Baseline correction
    baseline_idx = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    baseline_mean = np.mean(epoch[baseline_idx])
    baseline_std = np.std(epoch[baseline_idx])
    
    corrected = epoch - baseline_mean
    
    # 2. Calculate AMPLITUDE (peak-to-peak in MEP window)
    mep_idx = (time_vector >= mep_window[0]) & (time_vector <= mep_window[1])
    mep_signal = corrected[mep_idx]
    mep_times = time_vector[mep_idx]
    
    if len(mep_signal) == 0:
        return None
    
    amplitude = np.max(mep_signal) - np.min(mep_signal)
    
    # 3. Calculate LATENCY
    if latency_method == 'threshold':
        latency = calculate_mep_latency_threshold(epoch, time_vector, 
                                                  baseline_window=baseline_window)
    else:
        latency = None
    
    # 4. Find peak time (for visualization)
    peak_idx = np.argmax(np.abs(mep_signal))
    peak_time = mep_times[peak_idx]
    
    # 5. Validate MEP (must be >3 SDs above baseline)
    valid = amplitude > 3 * baseline_std
    
    return {
        'amplitude': amplitude,
        'latency': latency,
        'peak_time': peak_time,
        'baseline_std': baseline_std,
        'valid': valid
    }

# =============================================================================
# NEW: EXTRACT MEP DATA WITH LATENCY
# =============================================================================

def extract_mep_data_with_latency(emg_data, stimulus_times, chan_numbers, fs, 
                                  config, pre_ms=50, post_ms=150):
    """
    Extract MEP amplitudes AND latencies for all channels and stimuli
    
    Returns:
        DataFrame with columns: channel, hemisphere, amplitude, latency, etc.
    """
    pre_samples = int(pre_ms * fs / 1000)
    post_samples = int(post_ms * fs / 1000)
    time_vector = np.arange(-pre_samples, post_samples) / fs * 1000
    
    results = []
    switch_sample = int(config['hemisphere_switch_time'] * fs)
    
    print(f"\n📊 Extracting MEP features (amplitude + latency)...")
    
    for stim_idx, stim_time in enumerate(stimulus_times):
        # Check epoch boundaries
        if stim_time - pre_samples < 0 or stim_time + post_samples >= emg_data.shape[1]:
            continue
        
        # Determine hemisphere
        hemisphere = 'healthy' if stim_time < switch_sample else 'stroke'
        
        # Extract epoch for all channels
        epoch = emg_data[:, stim_time - pre_samples:stim_time + post_samples]
        
        # Process each channel
        for ch_idx, chan_num in enumerate(chan_numbers):
            if ch_idx >= epoch.shape[0]:
                continue
            
            # Calculate MEP features
            features = calculate_mep_features(epoch[ch_idx, :], time_vector)
            
            if features and features['valid']:
                results.append({
                    'channel': chan_num,
                    'hemisphere': hemisphere,
                    'stimulus_time': stim_time / fs,
                    'amplitude': features['amplitude'],
                    'latency': features['latency'],
                    'peak_time': features['peak_time'],
                    'baseline_std': features['baseline_std']
                })
    
    return pd.DataFrame(results)

# =============================================================================
# NEW: ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_latency_by_hemisphere(df, config, chan_numbers):
    """Analyze latency differences between hemispheres"""
    
    print("\n" + "="*70)
    print("MEP AMPLITUDE & LATENCY ANALYSIS")
    print("="*70)
    
    for hemisphere in ['healthy', 'stroke']:
        hemi_config = config[hemisphere]
        print(f"\n🧠 {hemisphere.upper()} HEMISPHERE:")
        print(f"   {hemi_config['description'] if 'description' in hemi_config else ''}")
        
        for i, (ch, muscle_name) in enumerate(zip(hemi_config['channels'], 
                                                  hemi_config['channel_names'])):
            channel_data = df[(df['hemisphere'] == hemisphere) & (df['channel'] == ch)]
            
            if len(channel_data) > 0:
                # Amplitude stats
                amp_mean = channel_data['amplitude'].mean()
                amp_std = channel_data['amplitude'].std()
                amp_sem = amp_std / np.sqrt(len(channel_data))
                
                # Latency stats (excluding None values)
                latency_data = channel_data['latency'].dropna()
                if len(latency_data) > 0:
                    lat_mean = latency_data.mean()
                    lat_std = latency_data.std()
                    lat_sem = lat_std / np.sqrt(len(latency_data))
                    
                    print(f"\n   {muscle_name} (Ch{ch}):")
                    print(f"      Amplitude: {amp_mean:.1f} ± {amp_sem:.1f} µV (n={len(channel_data)})")
                    print(f"      Latency:   {lat_mean:.2f} ± {lat_sem:.2f} ms (n={len(latency_data)})")
                else:
                    print(f"\n   {muscle_name} (Ch{ch}):")
                    print(f"      Amplitude: {amp_mean:.1f} ± {amp_sem:.1f} µV (n={len(channel_data)})")
                    print(f"      Latency:   No valid latencies detected")

def compare_latencies(df, config):
    """Compare latencies between healthy and stroke hemispheres"""
    
    print("\n" + "="*70)
    print("HEALTHY vs STROKE LATENCY COMPARISON")
    print("="*70)
    
    muscle_names = config['healthy']['channel_names']
    healthy_channels = config['healthy']['channels']
    stroke_channels = config['stroke']['channels']
    
    for i, muscle in enumerate(muscle_names):
        h_ch = healthy_channels[i]
        s_ch = stroke_channels[i]
        
        healthy_data = df[(df['hemisphere'] == 'healthy') & (df['channel'] == h_ch)]
        stroke_data = df[(df['hemisphere'] == 'stroke') & (df['channel'] == s_ch)]
        
        h_lat = healthy_data['latency'].dropna()
        s_lat = stroke_data['latency'].dropna()
        
        if len(h_lat) > 0 and len(s_lat) > 0:
            delay = s_lat.mean() - h_lat.mean()
            delay_pct = (delay / h_lat.mean()) * 100
            
            # Statistical test
            from scipy import stats as scipy_stats
            stat, p_val = scipy_stats.mannwhitneyu(h_lat, s_lat)
            sig = "***" if p_val < 0.05 else "ns"
            
            print(f"\n{muscle}:")
            print(f"   Healthy: {h_lat.mean():.2f} ± {h_lat.std():.2f} ms")
            print(f"   Stroke:  {s_lat.mean():.2f} ± {s_lat.std():.2f} ms")
            print(f"   Delay:   {delay:.2f} ms ({delay_pct:+.1f}%) - {sig}")
            print(f"   p-value: {p_val:.4f}")

def plot_latency_comparison(df, config, output_dir):
    """Create latency comparison plot"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'MEP Latency: {config["description"]}', 
                 fontsize=16, fontweight='bold')
    
    muscle_names = config['healthy']['channel_names']
    healthy_channels = config['healthy']['channels']
    stroke_channels = config['stroke']['channels']
    
    for i, (ax, muscle) in enumerate(zip(axes, muscle_names)):
        h_ch = healthy_channels[i]
        s_ch = stroke_channels[i]
        
        healthy_lat = df[(df['hemisphere'] == 'healthy') & (df['channel'] == h_ch)]['latency'].dropna()
        stroke_lat = df[(df['hemisphere'] == 'stroke') & (df['channel'] == s_ch)]['latency'].dropna()
        
        # Plot data
        positions = [1, 2]
        data = [healthy_lat, stroke_lat]
        
        bp = ax.boxplot(data, positions=positions, widths=0.5,
                        patch_artist=True, showmeans=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('#4CAF50')
        bp['boxes'][1].set_facecolor('#EF5350')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Healthy', 'Stroke'])
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(muscle, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add n values
        ax.text(1, ax.get_ylim()[1]*0.95, f'n={len(healthy_lat)}', 
               ha='center', fontsize=10)
        ax.text(2, ax.get_ylim()[1]*0.95, f'n={len(stroke_lat)}', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / 'mep_latency_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Latency plot saved: {plot_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MEP Amplitude & Latency Detection')
    parser.add_argument('folder', type=Path, help='Session folder')
    parser.add_argument('--fs', type=float, default=2000, help='Sampling rate (Hz)')
    parser.add_argument('--config', type=str, help='Experiment config name')
    parser.add_argument('--channel', type=int, help='Channel for artifact detection')
    parser.add_argument('--output', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.folder
    args.output.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("MEP AMPLITUDE & LATENCY DETECTION")
    print("="*70)
    
    try:
        # Get configuration
        config = get_experiment_config(args.folder, args.config)
        
        # Load EMG data
        print("\n🔄 Loading EMG data...")
        ts, emg_scaled, chan_numbers = load_emg_data_flexible(args.folder)
        
        print(f"📊 Data shape: {emg_scaled.shape}")
        print(f"📊 Duration: {ts[-1]:.1f} seconds")
        
        # Find stimulus artifacts
        print(f"\n🔍 Detecting stimulus artifacts...")
        
        if args.channel and args.channel in chan_numbers:
            detection_ch_idx = chan_numbers.index(args.channel)
        else:
            # Auto-select channel
            dynamic_ranges = [np.max(emg_scaled[i, :]) - np.min(emg_scaled[i, :]) 
                            for i in range(emg_scaled.shape[0])]
            detection_ch_idx = np.argmax(dynamic_ranges)
        
        stimulus_times = find_stimulus_artifacts(emg_scaled[detection_ch_idx, :], args.fs)
        print(f"🎯 Found {len(stimulus_times)} stimulus artifacts")
        
        if len(stimulus_times) == 0:
            print("❌ No stimuli detected!")
            return
        
        # Extract MEP features (amplitude + latency)
        df = extract_mep_data_with_latency(emg_scaled, stimulus_times, chan_numbers, 
                                          args.fs, config)
        
        print(f"\n✅ Extracted {len(df)} valid MEPs")
        
        # Save results
        output_file = args.output / 'mep_results_with_latency.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        # Analyze results
        analyze_latency_by_hemisphere(df, config, chan_numbers)
        compare_latencies(df, config)
        
        # Create visualization
        plot_latency_comparison(df, config, args.output)
        
        print(f"\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()