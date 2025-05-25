#!/usr/bin/env python3
"""
Simple Configurable Hemisphere-Aware TMS Detection Script
Edit the EXPERIMENT_CONFIGS dictionary below to add new experiments
"""
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# EXPERIMENT CONFIGURATIONS - EDIT THIS SECTION FOR NEW EXPERIMENTS
# ============================================================================

EXPERIMENT_CONFIGS = {
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "hemisphere_switch_time": 690.0,
        "healthy": {
            "channels": [2, 4, 7],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 182,
            "description": "Healthy Left Hemisphere → Right Muscles (0-690s)"
        },
        "stroke": {
            "channels": [8, 9, 11],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 262,
            "description": "Stroke Right Hemisphere → Left Muscles (690s-end)"
        }
    },

    "Nov5_Chive": {
        "description": "Nov5 Chive Experiment",
        "hemisphere_switch_time": 857.0,  # Match your sheet (857 seconds)
        "healthy": {
            "channels": [4, 2, 8],  # ✅ Ch9=Right Upper, Ch2=Right Forearm, Ch8=Right Hand
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 115,
            "description": "Healthy Left Hemisphere → Right Muscles"
        },
        "stroke": {
            "channels": [9, 10, 7],  # ✅ Ch4=Left Upper, Ch10=Left Forearm, Ch7=Left Hand
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 172,  # 287-115 = 172
            "description": "Stroke Right Hemisphere → Left Muscles"
        }
    },

    "Nov5_Cheddar": {
        "description": "Nov5 Cheddar Experiment",
        "hemisphere_switch_time": 650,  # seconds
        "healthy": {
            "channels": [2, 4, 12],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 105,
            "description": "Healthy hemisphere description"
        },
        "stroke": {
            "channels": [14, 9, 13],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 205,
            "description": "Stroke hemisphere description"
        }
    }
}

# ============================================================================
# MAIN DETECTION CODE - NO NEED TO EDIT BELOW THIS LINE
# ============================================================================

def get_experiment_config(folder_path, config_name=None):
    """Get experiment configuration based on folder name or specified config"""
    folder = Path(folder_path)
    folder_name = folder.name
    
    # If specific config requested
    if config_name and config_name in EXPERIMENT_CONFIGS:
        config = EXPERIMENT_CONFIGS[config_name].copy()
        print(f"📋 Using specified config: {config_name}")
        return config
    
    # Try exact match
    if folder_name in EXPERIMENT_CONFIGS:
        config = EXPERIMENT_CONFIGS[folder_name].copy()
        print(f"📋 Found exact match config: {folder_name}")
        return config
    
    # Try partial matching
    for key in EXPERIMENT_CONFIGS:
        if key.lower() in folder_name.lower() or folder_name.lower() in key.lower():
            config = EXPERIMENT_CONFIGS[key].copy()
            print(f"📋 Found partial match config: {key} for folder {folder_name}")
            return config
    
    # Use first available as default
    default_key = list(EXPERIMENT_CONFIGS.keys())[0]
    config = EXPERIMENT_CONFIGS[default_key].copy()
    print(f"⚠️ No matching config found for '{folder_name}', using default: {default_key}")
    print(f"💡 Add a config for '{folder_name}' in EXPERIMENT_CONFIGS to customize settings")
    
    return config

def detect_available_channels(folder):
    """Detect which channels are available in the folder"""
    chan_files = list(folder.glob("chan*.mat"))
    available_channels = []
    
    for f in chan_files:
        import re
        match = re.search(r'chan(\d+)\.mat', f.name)
        if match:
            available_channels.append(int(match.group(1)))
    
    available_channels.sort()
    print(f"🔍 Detected channels: {available_channels}")
    return available_channels

def load_emg_data_flexible(folder: Path):
    """Load EMG data with flexible channel numbering"""
    ts_path = folder / "Timestamps.mat"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {ts_path}")
    
    ts = sio.loadmat(str(ts_path))['analogInputDataTime_s'].flatten()
    
    # Find all channel files
    chan_files = list(folder.glob("chan*.mat"))
    chan_info = []
    
    for f in chan_files:
        import re
        match = re.search(r'chan(\d+)\.mat', f.name)
        if match:
            chan_num = int(match.group(1))
            chan_info.append((chan_num, f))
    
    # Sort by channel number
    chan_info.sort(key=lambda x: x[0])
    chan_numbers = [x[0] for x in chan_info]
    
    # Load channel data
    n_chan = len(chan_info)
    emg = np.zeros((n_chan, len(ts)), dtype=np.float32)
    
    for i, (chan_num, file_path) in enumerate(chan_info):
        data = sio.loadmat(str(file_path))['chandata'].flatten()
        emg[i] = data.astype(np.float32)
    
    print(f"📊 Loaded channels: {chan_numbers}")
    return ts, emg, chan_numbers

def butterworth_filter_superior(emg, fs, lowcut=10, highcut=500, order=2):
    """Superior Butterworth filtering"""
    print(f"🔧 Applying {order}-order Butterworth filter: {lowcut}-{highcut} Hz")
    
    nyq = fs / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    
    b, a = sig.butter(order, [low, high], btype='band')
    filtered = sig.filtfilt(b, a, emg, axis=1)
    rectified = np.abs(filtered)
    
    print(f"✅ Filtered signal range: {filtered.min():.3f} to {filtered.max():.3f}")
    return filtered, rectified

def hemisphere_detection_simple(emg_rect, chan_numbers, ts, fs, config):
    """Simple hemisphere-aware pulse detection"""
    
    print("🎯 HEMISPHERE-AWARE PULSE DETECTION")
    print("=" * 45)
    
    all_detections = []
    hemisphere_results = {}
    
    # Get switch time from config
    switch_time = config['hemisphere_switch_time']
    switch_idx = np.argmin(np.abs(ts - switch_time))
    
    print(f"📅 Hemisphere switch at {switch_time}s (sample {switch_idx})")
    
    # Process each hemisphere period
    for hemi_name in ['healthy', 'stroke']:
        hemi_config = config[hemi_name]
        target_channels = hemi_config['channels']
        expected_pulses = hemi_config['expected_pulses']
        
        print(f"\n🧠 {hemi_config['description']}")
        print(f"   Target channels: {target_channels}")
        print(f"   Expected pulses: {expected_pulses}")
        
        # Define time segment
        if hemi_name == 'healthy':
            start_idx = 0
            end_idx = switch_idx
        else:  # stroke
            start_idx = switch_idx
            end_idx = len(ts)
        
        segment_ts = ts[start_idx:end_idx]
        segment_emg = emg_rect[:, start_idx:end_idx]
        
        print(f"   Time segment: {segment_ts[0]:.1f}s - {segment_ts[-1]:.1f}s")
        
        # Check for missing channels
        available_target_channels = [ch for ch in target_channels if ch in chan_numbers]
        missing_channels = [ch for ch in target_channels if ch not in chan_numbers]
        
        if missing_channels:
            print(f"   ⚠️ Missing channels: {missing_channels}")
        
        if not available_target_channels:
            print(f"   ❌ No target channels available for {hemi_name} hemisphere!")
            continue
        
        # Select best channel from available target channels
        best_result = select_best_channel_for_segment(
            segment_emg, chan_numbers, available_target_channels, fs, expected_pulses
        )
        
        # Get global indices (add start_idx offset)
        segment_detections = best_result['detected_indices']
        global_detections = segment_detections + start_idx
        
        hemisphere_results[hemi_name] = {
            'channel_idx': best_result['channel_idx'],
            'channel_num': best_result['channel_num'],
            'detected_indices': global_detections,
            'time_range': (segment_ts[0], segment_ts[-1]),
            'config': hemi_config
        }
        
        print(f"   ✅ Selected Ch{best_result['channel_num']}")
        print(f"   Final detection: {len(global_detections)} pulses")
        
        # Store detections with hemisphere labels
        for idx in global_detections:
            all_detections.append({
                'pulse_index': idx,
                'hemisphere': hemi_name,
                'channel_used': best_result['channel_num']
            })
    
    return all_detections, hemisphere_results

def select_best_channel_for_segment(segment_emg, chan_numbers, target_channels, fs, expected_pulses):
    """Select best channel for a specific time segment"""
    
    # Get indices of target channels in the data
    target_indices = []
    for target_chan in target_channels:
        if target_chan in chan_numbers:
            target_indices.append(chan_numbers.index(target_chan))
    
    if not target_indices:
        raise ValueError(f"No target channels found")
    
    # Analyze each target channel in this segment
    results = []
    
    for idx in target_indices:
        chan_num = chan_numbers[idx]
        signal = segment_emg[idx]
        
        # Optimize detection for this channel/segment
        optimized_crossings = optimize_detection_threshold(signal, fs, expected_pulses)
        
        # Calculate basic metrics
        count_error = abs(len(optimized_crossings) - expected_pulses)
        snr = calculate_snr(signal, optimized_crossings, fs)
        
        results.append({
            'channel_idx': idx,
            'channel_num': chan_num,
            'count_error': count_error,
            'snr': snr,
            'pulse_count': len(optimized_crossings),
            'detected_indices': optimized_crossings
        })
        
        print(f"   Ch {chan_num}: {len(optimized_crossings)} pulses, error={count_error}, SNR={snr:.2f}")
    
    # Sort by error first, then SNR
    results.sort(key=lambda x: (x['count_error'], -x['snr']))
    best_result = results[0]
    
    return best_result

def optimize_detection_threshold(signal, fs, expected_n):
    """Optimize detection threshold for specific signal segment"""
    
    isi_samples = int(1.0 * fs)  # 1 second ISI
    collapse_samples = int(0.010 * fs)  # 10ms collapse window
    
    best_count = float('inf')
    best_indices = []
    
    # Search for optimal threshold multiplier
    for mult in np.arange(3.0, 15.0, 0.25):
        threshold = np.median(signal) + mult * np.median(np.abs(signal - np.median(signal)))
        
        # Find threshold crossings
        crossings = np.where(np.diff(signal > threshold) == 1)[0]
        
        # Apply filtering
        if len(crossings) > 1:
            # Remove close pulses (collapse window)
            keep = [True]
            for i in range(1, len(crossings)):
                keep.append(crossings[i] - crossings[i-1] > collapse_samples)
            crossings = crossings[keep]
            
            # Remove ISI violations
            keep = [True]
            for i in range(1, len(crossings)):
                keep.append(crossings[i] - crossings[i-1] > isi_samples)
            crossings = crossings[keep]
        
        count = len(crossings)
        if abs(count - expected_n) < abs(best_count - expected_n):
            best_count = count
            best_indices = crossings
    
    return best_indices

def calculate_snr(signal, crossings, fs):
    """Calculate signal-to-noise ratio"""
    if len(crossings) == 0:
        return 0
    
    pre_samples = int(0.2 * fs)  # 200ms window
    p2p_amps = []
    
    for cross_idx in crossings:
        if cross_idx - pre_samples >= 0 and cross_idx + pre_samples < len(signal):
            window = signal[cross_idx - pre_samples:cross_idx + pre_samples + 1]
            p2p_amps.append(np.max(window) - np.min(window))
    
    if not p2p_amps:
        return 0
    
    mean_p2p = np.mean(p2p_amps)
    base_std = np.std(signal[:pre_samples]) if pre_samples < len(signal) else np.std(signal)
    
    return mean_p2p / base_std if base_std > 0 else 0

def extract_hemisphere_specific_meps(emg_filtered, all_detections, hemisphere_results, ts, fs):
    """Extract MEPs using hemisphere-appropriate channels"""
    
    print("\n📡 EXTRACTING HEMISPHERE-SPECIFIC MEPs")
    print("=" * 40)
    
    pre_samples = int(50 * fs / 1000)  # 50ms pre
    post_samples = int(150 * fs / 1000)  # 150ms post
    
    n_chan = emg_filtered.shape[0]
    valid_epochs = []
    detection_info = []
    
    for detection in all_detections:
        pulse_idx = detection['pulse_index']
        hemisphere = detection['hemisphere']
        channel_used = detection['channel_used']
        
        if pulse_idx - pre_samples >= 0 and pulse_idx + post_samples < emg_filtered.shape[1]:
            # Extract epoch from all channels
            epoch = emg_filtered[:, pulse_idx - pre_samples:pulse_idx + post_samples]
            valid_epochs.append(epoch)
            
            detection_info.append({
                'detection_idx': len(valid_epochs) - 1,
                'pulse_index': pulse_idx,
                'pulse_time_s': ts[pulse_idx],
                'hemisphere': hemisphere,
                'detection_channel': channel_used
            })
    
    if valid_epochs:
        epochs = np.array(valid_epochs)
    else:
        epochs = np.zeros((0, n_chan, pre_samples + post_samples))
    
    print(f"✅ Extracted {len(valid_epochs)} valid MEP epochs")
    
    time_vector = np.arange(-pre_samples, post_samples) / fs * 1000
    return epochs, time_vector, detection_info

def compute_mep_amplitudes(epochs, time_vector, detection_info, chan_numbers, fs, config):
    """Compute MEP amplitudes with hemisphere awareness"""
    
    print("\n📏 COMPUTING MEP AMPLITUDES")
    print("=" * 30)
    
    n_epochs, n_chan, n_samples = epochs.shape
    
    # MEP measurement parameters
    baseline_ms = 50
    mep_window_ms = [15, 50]
    
    stim_idx = np.argmin(np.abs(time_vector))
    samples_per_ms = fs / 1000
    
    mep_start = stim_idx + int(mep_window_ms[0] * samples_per_ms)
    mep_end = stim_idx + int(mep_window_ms[1] * samples_per_ms)
    mep_start = max(mep_start, stim_idx + 1)
    mep_end = min(mep_end, n_samples - 1)
    
    baseline_end = int(baseline_ms * samples_per_ms)
    
    print(f"MEP window: {mep_window_ms}ms, Baseline: {baseline_ms}ms")
    print(f"Processing {n_epochs} epochs")
    
    results = []
    
    for epoch_idx in range(min(n_epochs, len(detection_info))):
        epoch_info = detection_info[epoch_idx]
        hemisphere = epoch_info['hemisphere']
        
        # Get relevant channels for this hemisphere from config
        relevant_channels = config[hemisphere]['channels']
        
        epoch_result = {
            'detection_idx': epoch_idx,
            'pulse_index': epoch_info['pulse_index'],
            'pulse_time_s': epoch_info['pulse_time_s'],
            'hemisphere': hemisphere,
            'detection_channel': epoch_info['detection_channel']
        }
        
        # Compute amplitudes for all channels
        for i, chan_num in enumerate(chan_numbers):
            if i < n_chan:
                signal = epochs[epoch_idx, i, :]
                
                if np.all(signal == 0):
                    amplitude = np.nan
                else:
                    # Baseline correction
                    if baseline_end > 0 and baseline_end < len(signal):
                        baseline_mean = np.mean(signal[:baseline_end])
                        signal_corrected = signal - baseline_mean
                    else:
                        signal_corrected = signal
                    
                    # MEP amplitude (peak-to-peak in physiological window)
                    if mep_end <= len(signal_corrected):
                        mep_signal = signal_corrected[mep_start:mep_end]
                        if len(mep_signal) > 0:
                            amplitude = np.max(mep_signal) - np.min(mep_signal)
                        else:
                            amplitude = np.nan
                    else:
                        amplitude = np.nan
                
                epoch_result[f'ch{chan_num}_amplitude'] = amplitude
                epoch_result[f'ch{chan_num}_relevant'] = chan_num in relevant_channels
        
        results.append(epoch_result)
    
    print(f"✅ Computed amplitudes for {len(results)} epochs")
    return results

def create_summary_plots(results_df, config):
    """Create summary plots for hemisphere comparison"""
    
    print("\n📈 Creating summary plots...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'MEP Detection Results: {config["description"]}', fontsize=14, fontweight='bold')
    
    # Separate data by hemisphere
    healthy_data = results_df[results_df['hemisphere'] == 'healthy']
    stroke_data = results_df[results_df['hemisphere'] == 'stroke']
    
    # Plot 1: Detection timeline
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(healthy_data['pulse_time_s']/60, [1]*len(healthy_data), 
               alpha=0.6, s=15, color='green', label=f'Healthy (n={len(healthy_data)})')
    ax1.scatter(stroke_data['pulse_time_s']/60, [0]*len(stroke_data),
               alpha=0.6, s=15, color='red', label=f'Stroke (n={len(stroke_data)})')
    ax1.axvline(config['hemisphere_switch_time']/60, color='black', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Hemisphere')
    ax1.set_title('Detection Timeline')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Stroke', 'Healthy'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot hand muscle responses (last channel in each hemisphere)
    healthy_hand_ch = config['healthy']['channels'][-1]
    stroke_hand_ch = config['stroke']['channels'][-1]
    
    # Plot 2: Healthy hand muscle
    ax2 = plt.subplot(2, 3, 2)
    healthy_hand_col = f'ch{healthy_hand_ch}_amplitude'
    if healthy_hand_col in healthy_data.columns:
        amps = healthy_data[healthy_hand_col].dropna()
        if len(amps) > 0:
            ax2.hist(amps, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(amps.mean(), color='red', linestyle='--', 
                       label=f'Mean: {amps.mean():.1f}µV')
            ax2.set_xlabel('MEP Amplitude (µV)')
            ax2.set_ylabel('Count')
            ax2.set_title(f'{config["healthy"]["channel_names"][-1]}\n(Healthy)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stroke hand muscle
    ax3 = plt.subplot(2, 3, 3)
    stroke_hand_col = f'ch{stroke_hand_ch}_amplitude'
    if stroke_hand_col in stroke_data.columns:
        amps = stroke_data[stroke_hand_col].dropna()
        if len(amps) > 0:
            ax3.hist(amps, bins=15, alpha=0.7, color='red', edgecolor='black')
            ax3.axvline(amps.mean(), color='blue', linestyle='--',
                       label=f'Mean: {amps.mean():.1f}µV')
            ax3.set_xlabel('MEP Amplitude (µV)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'{config["stroke"]["channel_names"][-1]}\n(Stroke)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel usage
    ax4 = plt.subplot(2, 3, 4)
    healthy_channels = healthy_data['detection_channel'].value_counts()
    stroke_channels = stroke_data['detection_channel'].value_counts()
    
    all_channels = sorted(set(list(healthy_channels.index) + list(stroke_channels.index)))
    x = np.arange(len(all_channels))
    width = 0.35
    
    healthy_counts = [healthy_channels.get(ch, 0) for ch in all_channels]
    stroke_counts = [stroke_channels.get(ch, 0) for ch in all_channels]
    
    ax4.bar(x - width/2, healthy_counts, width, label='Healthy', color='green', alpha=0.7)
    ax4.bar(x + width/2, stroke_counts, width, label='Stroke', color='red', alpha=0.7)
    
    ax4.set_xlabel('Detection Channel')
    ax4.set_ylabel('Usage Count')
    ax4.set_title('Channel Usage')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Ch{ch}' for ch in all_channels])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Summary text
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = f"EXPERIMENT SUMMARY\n{'='*20}\n\n"
    summary_text += f"Config: {config['description']}\n"
    summary_text += f"Switch: {config['hemisphere_switch_time']}s\n\n"
    
    summary_text += f"HEALTHY HEMISPHERE:\n"
    summary_text += f"  Expected: {config['healthy']['expected_pulses']}\n"
    summary_text += f"  Detected: {len(healthy_data)}\n"
    summary_text += f"  Channels: {config['healthy']['channels']}\n\n"
    
    summary_text += f"STROKE HEMISPHERE:\n"
    summary_text += f"  Expected: {config['stroke']['expected_pulses']}\n"
    summary_text += f"  Detected: {len(stroke_data)}\n"
    summary_text += f"  Channels: {config['stroke']['channels']}\n"
    
    # Add hand muscle comparison
    if (healthy_hand_col in healthy_data.columns and 
        stroke_hand_col in stroke_data.columns):
        healthy_hand_amps = healthy_data[healthy_hand_col].dropna()
        stroke_hand_amps = stroke_data[stroke_hand_col].dropna()
        
        if len(healthy_hand_amps) > 0 and len(stroke_hand_amps) > 0:
            ratio = stroke_hand_amps.mean() / healthy_hand_amps.mean()
            summary_text += f"\nHAND RATIO: {ratio:.2f}\n"
            
            if ratio < 0.5:
                summary_text += "Significant impairment"
            elif ratio < 0.8:
                summary_text += "Moderate impairment"
            else:
                summary_text += "Preserved function"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Simple Configurable MEP Detection')
    parser.add_argument('folder', type=Path, help='Session folder')
    parser.add_argument('--fs', type=float, default=2000, help='Sampling rate')
    parser.add_argument('--config', type=str, help='Experiment config name')
    
    args = parser.parse_args()
    
    print("="*60)
    print("    SIMPLE CONFIGURABLE MEP DETECTION")
    print("="*60)
    
    try:
        # Get configuration
        config = get_experiment_config(args.folder, args.config)
        
        # Check available channels
        available_channels = detect_available_channels(args.folder)
        
        # Load data
        ts, emg_raw, chan_numbers = load_emg_data_flexible(args.folder)
        
        # Signal scaling
        raw_range = np.max(np.abs(emg_raw))
        if raw_range < 1:
            scale_factor = 1e3
            emg_scaled = emg_raw * scale_factor
            unit_label = "µV (scaled)"
        else:
            emg_scaled = emg_raw
            unit_label = "µV"
        
        print(f"📊 Signal scaling: {unit_label}")
        
        # Apply filtering
        emg_filtered, emg_rect = butterworth_filter_superior(emg_scaled, args.fs)
        
        # Hemisphere-aware detection
        all_detections, hemisphere_results = hemisphere_detection_simple(
            emg_rect, chan_numbers, ts, args.fs, config
        )
        
        # Extract MEPs
        epochs, time_vector, detection_info = extract_hemisphere_specific_meps(
            emg_filtered, all_detections, hemisphere_results, ts, args.fs
        )
        
        # Compute amplitudes
        amplitude_results = compute_mep_amplitudes(
            epochs, time_vector, detection_info, chan_numbers, args.fs, config
        )
        
        # Create results DataFrame
        results_df = pd.DataFrame(amplitude_results)
        
        # Create plots
        fig = create_summary_plots(results_df, config)
        
        # Save results
        output_file = args.folder / 'mep_results.csv'
        results_df.to_csv(output_file, index=False)
        
        # Print final summary
        print(f"\n" + "="*60)
        print("           DETECTION COMPLETE")
        print("="*60)
        
        healthy_data = results_df[results_df['hemisphere'] == 'healthy']
        stroke_data = results_df[results_df['hemisphere'] == 'stroke']
        
        print(f"\n📊 SUMMARY:")
        print(f"   Experiment: {config['description']}")
        print(f"   Switch time: {config['hemisphere_switch_time']}s")
        print(f"   Healthy: {len(healthy_data)} pulses (expected: {config['healthy']['expected_pulses']})")
        print(f"   Stroke: {len(stroke_data)} pulses (expected: {config['stroke']['expected_pulses']})")
        print(f"   Total: {len(results_df)} MEPs detected")
        
        print(f"\n💾 Results saved: {output_file}")
        print(f"✅ Analysis complete!")
        
        # Show available configurations
        print(f"\n💡 Available configurations:")
        for name in EXPERIMENT_CONFIGS.keys():
            print(f"   - {name}")
        print(f"   Edit EXPERIMENT_CONFIGS in the script to add new experiments")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()