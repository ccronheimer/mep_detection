#!/usr/bin/env python3
"""
MEP Raw Epoch Viewer
Reads MEP data and displays raw epoch examples from different channels and hemispheres
"""
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from pathlib import Path

# Import the experiment configurations from your detection script
EXPERIMENT_CONFIGS = {
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
        "description": "Oct5 Chive Experiment",
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
    },
    "Oct31_Cheddar": {
        "description": "Oct31 Cheddar Experiment",
        "hemisphere_switch_time": 500.0,
        "healthy": {
            "channels": [1, 2, 3],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 85,
            "description": "Healthy Left Hemisphere → Right Muscles"
        },
        "stroke": {
            "channels": [4, 5, 6],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 254,
            "description": "Stroke Right Hemisphere → Left Muscles"
        }
    }
}

def get_experiment_config(folder_path, config_name=None):
    """Get experiment configuration based on folder name or specified config"""
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

def find_stimulus_artifacts(signal, fs, min_amplitude_factor=5, min_isi_ms=800):
    """
    Find stimulus artifacts in EMG signal
    These appear as large amplitude spikes at regular intervals
    """
    # Calculate threshold based on signal statistics
    median_val = np.median(signal)
    mad = np.median(np.abs(signal - median_val))
    threshold = median_val + min_amplitude_factor * mad
    
    # Find threshold crossings (both positive and negative)
    pos_crossings = np.where(np.diff(signal > threshold) == 1)[0]
    neg_crossings = np.where(np.diff(signal < -threshold) == 1)[0]
    
    # Combine and sort all crossings
    all_crossings = np.sort(np.concatenate([pos_crossings, neg_crossings]))
    
    if len(all_crossings) == 0:
        return np.array([])
    
    # Filter out crossings that are too close together (enforce minimum ISI)
    min_isi_samples = int(min_isi_ms * fs / 1000)
    filtered_crossings = [all_crossings[0]]
    
    for crossing in all_crossings[1:]:
        if crossing - filtered_crossings[-1] > min_isi_samples:
            filtered_crossings.append(crossing)
    
    return np.array(filtered_crossings)

def extract_raw_epochs(emg_data, stimulus_times, fs, pre_ms=50, post_ms=150):
    """Extract raw epochs around stimulus times"""
    pre_samples = int(pre_ms * fs / 1000)
    post_samples = int(post_ms * fs / 1000)
    
    n_channels = emg_data.shape[0]
    valid_epochs = []
    valid_times = []
    
    for stim_time in stimulus_times:
        if stim_time - pre_samples >= 0 and stim_time + post_samples < emg_data.shape[1]:
            epoch = emg_data[:, stim_time - pre_samples:stim_time + post_samples]
            valid_epochs.append(epoch)
            valid_times.append(stim_time)
    
    if valid_epochs:
        epochs = np.array(valid_epochs)
    else:
        epochs = np.zeros((0, n_channels, pre_samples + post_samples))
    
    time_vector = np.arange(-pre_samples, post_samples) / fs * 1000  # in ms
    
    return epochs, time_vector, valid_times

def plot_raw_epoch_examples(epochs, time_vector, config, chan_numbers, 
                          hemisphere_epochs, ts, fs, n_examples=3):
    """Plot examples of raw MEP epochs"""
    
    if len(epochs) == 0:
        print("❌ No epochs found to plot!")
        return
    
    # Create larger figure with better spacing
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Raw MEP Epoch Examples: {config["description"]}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Determine layout - use fewer columns for better readability
    n_channels = len(chan_numbers)
    n_cols = min(2, n_examples)  # Maximum 2 columns for better visibility
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    # Select example epochs
    healthy_indices = hemisphere_epochs.get('healthy', [])
    stroke_indices = hemisphere_epochs.get('stroke', [])
    
    # Pick some examples from each hemisphere
    examples = []
    
    if healthy_indices:
        healthy_examples = np.random.choice(healthy_indices, 
                                          min(n_examples//2, len(healthy_indices)), 
                                          replace=False)
        for idx in healthy_examples:
            examples.append(('healthy', idx))
    
    if stroke_indices:
        stroke_examples = np.random.choice(stroke_indices, 
                                         min(n_examples//2, len(stroke_indices)), 
                                         replace=False)
        for idx in stroke_examples:
            examples.append(('stroke', idx))
    
    # If we don't have enough from both hemispheres, fill with any available
    if len(examples) < n_examples:
        all_indices = list(range(len(epochs)))
        remaining_needed = n_examples - len(examples)
        used_indices = [ex[1] for ex in examples]
        available_indices = [i for i in all_indices if i not in used_indices]
        
        if available_indices:
            additional = np.random.choice(available_indices, 
                                        min(remaining_needed, len(available_indices)), 
                                        replace=False)
            for idx in additional:
                # Determine hemisphere for this epoch
                if idx in healthy_indices:
                    hemi = 'healthy'
                elif idx in stroke_indices:
                    hemi = 'stroke'
                else:
                    hemi = 'unknown'
                examples.append((hemi, idx))
    
    # Plot examples
    for plot_idx, (hemisphere, epoch_idx) in enumerate(examples[:n_examples]):
        if plot_idx >= n_examples:
            break
            
        epoch = epochs[epoch_idx]
        
        # Determine relevant channels for this hemisphere
        if hemisphere in config:
            relevant_channels = config[hemisphere]['channels']
            hemi_description = config[hemisphere]['description']
        else:
            relevant_channels = []
            hemi_description = "Unknown hemisphere"
        
        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        
        # Plot all channels, highlight relevant ones
        legend_entries = []
        for ch_idx, chan_num in enumerate(chan_numbers):
            if ch_idx < epoch.shape[0]:
                signal = epoch[ch_idx, :]
                
                if chan_num in relevant_channels:
                    # Highlight relevant channels with thicker lines and bright colors
                    color = 'red' if hemisphere == 'stroke' else 'green'
                    linewidth = 3
                    alpha = 1.0
                    ch_name_idx = relevant_channels.index(chan_num)
                    if hemisphere in config:
                        label = f"Ch{chan_num}: {config[hemisphere]['channel_names'][ch_name_idx]}"
                    else:
                        label = f"Ch{chan_num}"
                    
                    ax.plot(time_vector, signal, color=color, linewidth=linewidth, 
                           alpha=alpha, label=label)
                    legend_entries.append(label)
                else:
                    # Non-relevant channels in lighter gray
                    color = 'lightgray'
                    linewidth = 1
                    alpha = 0.6
                    
                    ax.plot(time_vector, signal, color=color, linewidth=linewidth, 
                           alpha=alpha)
        
        # Add stimulus line with more prominent styling
        ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=3, 
                  label='TMS Stimulus')
        
        # Highlight MEP window with more visible shading
        ax.axvspan(15, 50, alpha=0.3, color='yellow', label='MEP Window (15-50ms)')
        
        # Improved formatting
        ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=14, fontweight='bold')
        ax.set_title(f'Example {plot_idx + 1}: {hemisphere.title()} Hemisphere\n'
                    f'{hemi_description}', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.4, linewidth=1)
        
        # Larger tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Create legend with only relevant channels plus stimulus markers
        legend_labels = legend_entries + ['TMS Stimulus', 'MEP Window (15-50ms)']
        handles = ax.get_lines()[-len(legend_labels):]  # Get the last few handles
        
        # Create custom legend handles
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        
        legend_handles = []
        for entry in legend_entries:
            if 'stroke' in hemisphere.lower():
                color = 'red'
            else:
                color = 'green'
            legend_handles.append(Line2D([0], [0], color=color, linewidth=3, label=entry))
        
        legend_handles.append(Line2D([0], [0], color='black', linewidth=3, 
                                   linestyle='--', label='TMS Stimulus'))
        legend_handles.append(Rectangle((0,0),1,1, facecolor='yellow', alpha=0.3, 
                                      label='MEP Window'))
        
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', 
                 fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout with more spacing
    plt.tight_layout(rect=[0, 0, 0.85, 0.96], h_pad=3.0, w_pad=2.0)
    plt.show()
    
    return fig

def analyze_signal_quality(emg_data, chan_numbers, fs):
    """Analyze signal quality across channels"""
    print("\n📊 SIGNAL QUALITY ANALYSIS")
    print("=" * 40)
    
    for i, chan_num in enumerate(chan_numbers):
        if i < emg_data.shape[0]:
            signal = emg_data[i, :]
            
            # Basic statistics
            rms = np.sqrt(np.mean(signal**2))
            peak_to_peak = np.max(signal) - np.min(signal)
            snr_estimate = np.mean(np.abs(signal)) / np.std(signal)
            
            print(f"Ch{chan_num:2d}: RMS={rms:8.2f}µV, "
                  f"P2P={peak_to_peak:8.2f}µV, SNR≈{snr_estimate:.2f}")

def main():
    parser = argparse.ArgumentParser(description='MEP Raw Epoch Viewer')
    parser.add_argument('folder', type=Path, help='Session folder')
    parser.add_argument('--fs', type=float, default=2000, help='Sampling rate (Hz)')
    parser.add_argument('--config', type=str, help='Experiment config name')
    parser.add_argument('--examples', type=int, default=3, help='Number of example epochs to show')
    parser.add_argument('--channel', type=int, help='Focus on specific channel for artifact detection')
    
    args = parser.parse_args()
    
    print("="*60)
    print("         MEP RAW EPOCH VIEWER")
    print("="*60)
    
    try:
        # Get configuration
        config = get_experiment_config(args.folder, args.config)
        
        # Load data
        print("\n🔄 Loading EMG data...")
        ts, emg_raw, chan_numbers = load_emg_data_flexible(args.folder)
        
        # Signal scaling for better visualization
        raw_range = np.max(np.abs(emg_raw))
        if raw_range < 1:
            scale_factor = 1e6  # Convert to µV if data is in V
            emg_scaled = emg_raw * scale_factor
            unit_label = "µV (scaled from V)"
        elif raw_range < 1000:
            scale_factor = 1e3  # Convert to µV if data is in mV
            emg_scaled = emg_raw * scale_factor
            unit_label = "µV (scaled from mV)"
        else:
            emg_scaled = emg_raw
            unit_label = "µV"
        
        print(f"📊 Signal scaling: {unit_label}")
        print(f"📊 Data shape: {emg_scaled.shape} (channels × samples)")
        print(f"📊 Duration: {ts[-1]:.1f} seconds")
        
        # Analyze signal quality
        analyze_signal_quality(emg_scaled, chan_numbers, args.fs)
        
        # Find stimulus artifacts
        print(f"\n🔍 Detecting stimulus artifacts...")
        
        # Use specified channel or find best channel for artifact detection
        if args.channel and args.channel in chan_numbers:
            detection_ch_idx = chan_numbers.index(args.channel)
            print(f"Using specified channel {args.channel} for artifact detection")
        else:
            # Find channel with highest dynamic range (likely to have clear artifacts)
            dynamic_ranges = []
            for i in range(emg_scaled.shape[0]):
                signal = emg_scaled[i, :]
                dynamic_ranges.append(np.max(signal) - np.min(signal))
            
            detection_ch_idx = np.argmax(dynamic_ranges)
            detection_channel = chan_numbers[detection_ch_idx]
            print(f"Auto-selected channel {detection_channel} for artifact detection "
                  f"(highest dynamic range: {dynamic_ranges[detection_ch_idx]:.1f}µV)")
        
        # Detect stimulus artifacts
        stimulus_times = find_stimulus_artifacts(emg_scaled[detection_ch_idx, :], args.fs)
        
        print(f"🎯 Found {len(stimulus_times)} potential stimulus artifacts")
        
        if len(stimulus_times) == 0:
            print("❌ No stimulus artifacts detected!")
            print("💡 Try adjusting detection parameters or check signal quality")
            return
        
        # Separate epochs by hemisphere based on timing
        switch_time = config['hemisphere_switch_time']
        switch_sample = int(switch_time * args.fs)
        
        healthy_epoch_indices = []
        stroke_epoch_indices = []
        hemisphere_epochs = {}
        
        for i, stim_time in enumerate(stimulus_times):
            if stim_time < switch_sample:
                healthy_epoch_indices.append(i)
            else:
                stroke_epoch_indices.append(i)
        
        hemisphere_epochs['healthy'] = healthy_epoch_indices
        hemisphere_epochs['stroke'] = stroke_epoch_indices
        
        print(f"📊 Hemisphere distribution:")
        print(f"   Healthy (0-{switch_time}s): {len(healthy_epoch_indices)} stimuli")
        print(f"   Stroke ({switch_time}s-end): {len(stroke_epoch_indices)} stimuli")
        
        # Extract raw epochs
        print(f"\n📡 Extracting raw epochs...")
        epochs, time_vector, valid_stimulus_times = extract_raw_epochs(
            emg_scaled, stimulus_times, args.fs
        )
        
        print(f"✅ Extracted {len(epochs)} valid epochs")
        print(f"📏 Epoch duration: {time_vector[0]:.1f} to {time_vector[-1]:.1f} ms")
        
        # Plot examples
        print(f"\n📈 Creating epoch plots...")
        fig = plot_raw_epoch_examples(
            epochs, time_vector, config, chan_numbers, 
            hemisphere_epochs, ts, args.fs, args.examples
        )
        
        # Print summary
        print(f"\n" + "="*60)
        print("           EPOCH VIEWING COMPLETE")
        print("="*60)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Experiment: {config['description']}")
        print(f"   Total epochs: {len(epochs)}")
        print(f"   Healthy epochs: {len(healthy_epoch_indices)}")
        print(f"   Stroke epochs: {len(stroke_epoch_indices)}")
        print(f"   Detection channel: Ch{chan_numbers[detection_ch_idx]}")
        
        # Show channel assignments
        print(f"\n🧠 CHANNEL ASSIGNMENTS:")
        print(f"   Healthy hemisphere ({config['healthy']['description']}):")
        for i, (ch, name) in enumerate(zip(config['healthy']['channels'], 
                                         config['healthy']['channel_names'])):
            print(f"      Ch{ch}: {name}")
        
        print(f"   Stroke hemisphere ({config['stroke']['description']}):")
        for i, (ch, name) in enumerate(zip(config['stroke']['channels'], 
                                         config['stroke']['channel_names'])):
            print(f"      Ch{ch}: {name}")
        
        print(f"\n✅ Visualization complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()