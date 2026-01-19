#!/usr/bin/env python3
"""
MEP Waveform Morphology Analysis
Analyzes MEP waveform characteristics using your existing data structure
NOW WITH ARTIFACT REJECTION

Usage:
    python mep_morphology_analysis.py ./
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy import signal, stats
from scipy.signal import find_peaks
import argparse

# =============================================================================
# CONFIGURATION - MATCHES YOUR EXISTING SCRIPTS
# =============================================================================

ANALYSIS_CONFIGS = {
    "NHP1": {
        "folder": "Nov5_Olive",
        "color": "#FF99B4",
        "hemisphere_switch_time": 695.0,
        "healthy_channels": [2, 4, 7],
        "stroke_channels": [8, 9, 11],
    },
    "NHP2": {
        "folder": "Nov5_Cheddar",
        "color": "#FFB3C6",
        "hemisphere_switch_time": 650.0,
        "healthy_channels": [2, 4, 12],
        "stroke_channels": [14, 9, 13],
    },
    "NHP3": {
        "folder": "Oct31_Chive",
        "color": "#7DD3C0",
        "hemisphere_switch_time": 930.0,
        "healthy_channels": [1, 2, 3],
        "stroke_channels": [4, 5, 6],
    }
}

MUSCLE_MAPPING = {0: "Biceps", 1: "Brach", 2: "APB"}

# =============================================================================
# ARTIFACT REJECTION
# =============================================================================

def is_valid_epoch(epoch, time_vector, max_amplitude=500, min_std=0.05):
    """
    Reject artifacts and bad epochs
    
    Parameters:
    -----------
    epoch : array
        Single MEP waveform
    time_vector : array
        Time points
    max_amplitude : float
        Maximum allowed amplitude (µV) - rejects huge artifacts
    min_std : float
        Minimum standard deviation - rejects flat/dead channels (lowered to 0.05 for weak MEPs)
    
    Returns:
    --------
    bool : True if epoch is valid
    """
    # 1. Check for extreme amplitudes (artifacts, movement, saturation)
    max_val = np.max(np.abs(epoch))
    if max_val > max_amplitude:
        return False
    
    # 2. Check for flat/dead channels (very permissive for weak signals)
    epoch_std = np.std(epoch)
    if epoch_std < min_std:
        return False
    
    # 3. Check for saturated signals (many samples at extreme values)
    saturation_threshold = 400  # µV
    n_saturated = np.sum(np.abs(epoch) > saturation_threshold)
    if n_saturated > 10:  # More than 10 samples saturated
        return False
    
    # 4. Check for excessive discontinuities (electrical noise)
    diff = np.diff(epoch)
    if np.max(np.abs(diff)) > 300:  # Single-sample jump > 300 µV
        return False
    
    return True

# =============================================================================
# DATA LOADING - USES YOUR EXISTING STRUCTURE
# =============================================================================

def load_channel_data(folder, channel_num):
    """Load data from a single channel .mat file"""
    filepath = folder / f"chan{channel_num}.mat"
    
    if not filepath.exists():
        return None
    
    try:
        data = sio.loadmat(str(filepath))
        return data['chandata'].flatten()
    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return None

def find_stimulus_artifacts(signal, fs, min_amplitude_factor=5, min_isi_ms=800):
    """Find stimulus artifacts - same as your epoch viewer"""
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

def extract_channel_epochs(channel_data, stimulus_times, fs, pre_ms=50, post_ms=150):
    """Extract epochs from a single channel"""
    pre_samples = int(pre_ms * fs / 1000)
    post_samples = int(post_ms * fs / 1000)
    
    epochs = []
    for stim_time in stimulus_times:
        if stim_time - pre_samples >= 0 and stim_time + post_samples < len(channel_data):
            epoch = channel_data[stim_time - pre_samples:stim_time + post_samples]
            epochs.append(epoch)
    
    time_vector = np.arange(-pre_samples, post_samples) / fs * 1000
    
    return np.array(epochs), time_vector

# =============================================================================
# WAVEFORM FEATURE EXTRACTION
# =============================================================================

def extract_waveform_features(epoch, time_vector, baseline_window=(-50, -10), 
                              mep_window=(5, 50), fs=2000):
    """Extract morphological features from a single MEP waveform"""
    
    baseline_idx = (time_vector >= baseline_window[0]) & (time_vector <= baseline_window[1])
    if np.sum(baseline_idx) == 0:
        return None
    
    baseline_mean = np.mean(epoch[baseline_idx])
    baseline_std = np.std(epoch[baseline_idx])
    
    if baseline_std == 0:
        return None
    
    corrected = epoch - baseline_mean
    
    mep_idx = (time_vector >= mep_window[0]) & (time_vector <= mep_window[1])
    mep_signal = corrected[mep_idx]
    mep_times = time_vector[mep_idx]
    
    if len(mep_signal) == 0:
        return None
    
    rectified = np.abs(mep_signal)
    
    # Amplitude features
    peak_to_peak = np.max(mep_signal) - np.min(mep_signal)
    peak_idx = np.argmax(rectified)
    peak_amplitude = rectified[peak_idx]
    onset_latency = mep_times[peak_idx]
    
    # Temporal dispersion
    threshold = baseline_std * 3
    above_threshold = rectified > threshold
    
    if np.any(above_threshold):
        onset_idx = np.argmax(above_threshold)
        offset_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
        mep_duration = mep_times[offset_idx] - mep_times[onset_idx]
        onset_time = mep_times[onset_idx]
        offset_time = mep_times[offset_idx]
    else:
        mep_duration = 0
        onset_time = np.nan
        offset_time = np.nan
    
    # Rise and fall time
    if not np.isnan(onset_time):
        rise_time = onset_latency - onset_time
        fall_time = offset_time - onset_latency
        rise_fall_ratio = rise_time / fall_time if fall_time > 0 else np.nan
    else:
        rise_time = np.nan
        fall_time = np.nan
        rise_fall_ratio = np.nan
    
    # Fragmentation
    peak_threshold = threshold * 2
    peaks, _ = find_peaks(rectified, height=peak_threshold, distance=int(fs * 0.002))
    num_peaks = len(peaks)
    is_polyphasic = num_peaks > 2
    
    # Desynchronization
    auc = np.trapz(rectified, mep_times)
    zero_crossings = np.sum(np.diff(np.sign(mep_signal)) != 0)
    signal_diff = np.diff(rectified)
    smoothness = np.std(signal_diff) if len(signal_diff) > 0 else np.nan
    
    # Shape descriptors
    skewness = stats.skew(mep_signal)
    kurtosis = stats.kurtosis(mep_signal)
    
    return {
        'peak_to_peak_amplitude': peak_to_peak,
        'peak_amplitude': peak_amplitude,
        'onset_latency': onset_latency,
        'auc': auc,
        'mep_duration': mep_duration,
        'onset_time': onset_time,
        'offset_time': offset_time,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'rise_fall_ratio': rise_fall_ratio,
        'num_peaks': num_peaks,
        'is_polyphasic': is_polyphasic,
        'zero_crossings': zero_crossings,
        'smoothness': smoothness,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

# =============================================================================
# DATA LOADING AND ORGANIZATION
# =============================================================================

def load_and_organize_data(base_path: Path, fs=2000, max_amplitude=500):
    """Load waveform data organized by subject, muscle, hemisphere"""
    all_data = {}
    
    for subject, config in ANALYSIS_CONFIGS.items():
        folder = base_path / config['folder']
        
        if not folder.exists():
            print(f"⚠️  Skipping {subject}: Folder not found ({folder})")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {subject} ({config['folder']})")
        print(f"{'='*70}")
        
        # Load timestamps
        ts_path = folder / "Timestamps.mat"
        if not ts_path.exists():
            print(f"⚠️  Skipping {subject}: No Timestamps.mat")
            continue
        
        ts = sio.loadmat(str(ts_path))['analogInputDataTime_s'].flatten()
        switch_sample = int(config['hemisphere_switch_time'] * fs)
        
        subject_data = {}
        
        for muscle_idx, muscle_name in MUSCLE_MAPPING.items():
            healthy_ch = config['healthy_channels'][muscle_idx]
            stroke_ch = config['stroke_channels'][muscle_idx]
            
            print(f"\n  {muscle_name}:")
            print(f"    Healthy Ch{healthy_ch}, Stroke Ch{stroke_ch}")
            
            # Load channel data
            healthy_data = load_channel_data(folder, healthy_ch)
            stroke_data = load_channel_data(folder, stroke_ch)
            
            if healthy_data is None or stroke_data is None:
                print(f"    ⚠️  Could not load channel data")
                continue
            
            # Detect stimuli on healthy channel (usually cleaner)
            stimulus_times = find_stimulus_artifacts(healthy_data, fs)
            print(f"    Found {len(stimulus_times)} stimulus artifacts")
            
            if len(stimulus_times) == 0:
                print(f"    ⚠️  No stimuli detected")
                continue
            
            # Separate by hemisphere
            healthy_stim_times = stimulus_times[stimulus_times < switch_sample]
            stroke_stim_times = stimulus_times[stimulus_times >= switch_sample]
            
            print(f"    Healthy: {len(healthy_stim_times)} epochs")
            print(f"    Stroke: {len(stroke_stim_times)} epochs")
            
            # Extract epochs
            healthy_epochs, time_vector = extract_channel_epochs(healthy_data, healthy_stim_times, fs)
            stroke_epochs, _ = extract_channel_epochs(stroke_data, stroke_stim_times, fs)
            
            # Extract features WITH ARTIFACT REJECTION
            healthy_features = []
            healthy_epochs_clean = []
            rejected_healthy = 0
            
            for i, epoch in enumerate(healthy_epochs):
                # CHECK FOR ARTIFACTS
                if not is_valid_epoch(epoch, time_vector, max_amplitude=max_amplitude):
                    rejected_healthy += 1
                    continue
                
                features = extract_waveform_features(epoch, time_vector, fs=fs)
                if features is not None:
                    features['trial'] = i
                    healthy_features.append(features)
                    healthy_epochs_clean.append(epoch)
            
            stroke_features = []
            stroke_epochs_clean = []
            rejected_stroke = 0
            
            for i, epoch in enumerate(stroke_epochs):
                # CHECK FOR ARTIFACTS
                if not is_valid_epoch(epoch, time_vector, max_amplitude=max_amplitude):
                    rejected_stroke += 1
                    continue
                
                features = extract_waveform_features(epoch, time_vector, fs=fs)
                if features is not None:
                    features['trial'] = i
                    stroke_features.append(features)
                    stroke_epochs_clean.append(epoch)
            
            print(f"    Features extracted: {len(healthy_features)} healthy, {len(stroke_features)} stroke")
            print(f"    Artifacts rejected: {rejected_healthy} healthy, {rejected_stroke} stroke")
            
            subject_data[muscle_name] = {
                'healthy': pd.DataFrame(healthy_features) if healthy_features else pd.DataFrame(),
                'stroke': pd.DataFrame(stroke_features) if stroke_features else pd.DataFrame(),
                'healthy_epochs': np.array(healthy_epochs_clean),
                'stroke_epochs': np.array(stroke_epochs_clean),
                'time': time_vector,
                'color': config['color']
            }
        
        all_data[subject] = subject_data
        print(f"\n✓ Loaded {subject}")
    
    return all_data

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_waveform_overlays(all_data, output_dir, n_trials=30):
    """Plot overlaid waveforms"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('MEP Waveform Morphology: Individual Trials (Artifacts Removed)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    subjects = list(all_data.keys())
    muscles = list(MUSCLE_MAPPING.values())
    
    for row, subject in enumerate(subjects):
        for col, muscle in enumerate(muscles):
            ax = axes[row, col]
            
            if subject not in all_data or muscle not in all_data[subject]:
                ax.axis('off')
                continue
            
            data = all_data[subject][muscle]
            time = data['time']
            healthy_epochs = data['healthy_epochs']
            stroke_epochs = data['stroke_epochs']
            
            if len(healthy_epochs) == 0 and len(stroke_epochs) == 0:
                ax.axis('off')
                continue
            
            # Plot sample of trials
            n_plot = min(n_trials, len(healthy_epochs), len(stroke_epochs))
            
            for i in range(min(n_plot, len(healthy_epochs))):
                ax.plot(time, healthy_epochs[i], 'g-', alpha=0.2, linewidth=0.5)
            
            for i in range(min(n_plot, len(stroke_epochs))):
                ax.plot(time, stroke_epochs[i], 'r-', alpha=0.2, linewidth=0.5)
            
            # Plot means
            if len(healthy_epochs) > 0:
                ax.plot(time, np.mean(healthy_epochs[:n_plot], axis=0), 
                       'g-', linewidth=2.5, label='Healthy Mean')
            if len(stroke_epochs) > 0:
                ax.plot(time, np.mean(stroke_epochs[:n_plot], axis=0), 
                       'r-', linewidth=2.5, label='Stroke Mean')
            
            # Formatting
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (ms)', fontsize=10)
            ax.set_ylabel('Amplitude (µV)', fontsize=10)
            ax.set_title(f'{subject} - {muscle}', fontsize=12, fontweight='bold')
            ax.set_xlim([-10, 60])
            ax.legend(loc='upper right', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plot_path = output_dir / 'waveform_overlays.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_morphology_comparison(all_data, output_dir):
    """Create morphology feature comparison"""
    
    features_to_plot = [
        ('mep_duration', 'MEP Duration (ms)', 'Temporal\nDispersion'),
        ('rise_time', 'Rise Time (ms)', 'Synchronization'),
        ('fall_time', 'Fall Time (ms)', 'Synchronization'),
        ('rise_fall_ratio', 'Rise/Fall Ratio', 'Asymmetry'),
        ('num_peaks', 'Number of Peaks', 'Fragmentation'),
        ('zero_crossings', 'Zero Crossings', 'Complexity'),
        ('smoothness', 'Signal Roughness', 'Desynchronization'),
        ('peak_amplitude', 'Peak Amplitude (µV)', 'Excitability'),
        ('auc', 'Area Under Curve', 'Total Activation'),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('MEP Morphology Features: Healthy vs Stroke', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    subjects = list(all_data.keys())
    muscles = list(MUSCLE_MAPPING.values())
    
    for idx, (feature, ylabel, category) in enumerate(features_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        all_healthy = []
        all_stroke = []
        positions_h = []
        positions_s = []
        labels = []
        
        pos = 0
        for subject in subjects:
            for muscle in muscles:
                if subject not in all_data or muscle not in all_data[subject]:
                    continue
                
                data = all_data[subject][muscle]
                df_h = data['healthy']
                df_s = data['stroke']
                
                if feature in df_h.columns and feature in df_s.columns:
                    healthy_vals = df_h[feature].dropna()
                    stroke_vals = df_s[feature].dropna()
                    
                    if len(healthy_vals) > 0:
                        all_healthy.append(healthy_vals)
                        positions_h.append(pos)
                        labels.append(f'{subject}\n{muscle}')
                        pos += 1
                    
                    if len(stroke_vals) > 0:
                        all_stroke.append(stroke_vals)
                        positions_s.append(pos)
                        pos += 1
                        pos += 0.5
        
        # Plot violins
        if all_healthy:
            parts_h = ax.violinplot(all_healthy, positions=positions_h, 
                                   widths=0.7, showmeans=True)
            for pc in parts_h['bodies']:
                pc.set_facecolor('green')
                pc.set_alpha(0.3)
        
        if all_stroke:
            parts_s = ax.violinplot(all_stroke, positions=positions_s,
                                   widths=0.7, showmeans=True)
            for pc in parts_s['bodies']:
                pc.set_facecolor('#FF6B6B')
                pc.set_alpha(0.3)
        
        # Formatting
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(category, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_xticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plot_path = output_dir / 'morphology_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {plot_path}")
    plt.close()

def plot_fragmentation_summary(all_data, output_dir):
    """Summary of fragmentation analysis"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Fragmentation Analysis: Polyphasic Responses', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    subjects = list(all_data.keys())
    muscles = list(MUSCLE_MAPPING.values())
    
    for row, subject in enumerate(subjects):
        for col, muscle in enumerate(muscles):
            ax = axes[row, col]
            
            if subject not in all_data or muscle not in all_data[subject]:
                ax.axis('off')
                continue
            
            data = all_data[subject][muscle]
            df_h = data['healthy']
            df_s = data['stroke']
            
            if len(df_h) == 0 and len(df_s) == 0:
                ax.axis('off')
                continue
            
            # Calculate polyphasic percentages
            if 'is_polyphasic' in df_h.columns and 'is_polyphasic' in df_s.columns:
                h_poly = df_h['is_polyphasic'].sum() / len(df_h) * 100 if len(df_h) > 0 else 0
                s_poly = df_s['is_polyphasic'].sum() / len(df_s) * 100 if len(df_s) > 0 else 0
                
                bars = ax.bar(['Healthy', 'Stroke'], [h_poly, s_poly],
                             color=['green', '#FF6B6B'], alpha=0.6)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', 
                           fontweight='bold', fontsize=9)
                
                ax.set_ylabel('Polyphasic (%)', fontsize=10)
                ax.set_title(f'{subject} - {muscle}', fontsize=11, fontweight='bold')
                ax.set_ylim([0, 100])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plot_path = output_dir / 'fragmentation_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {plot_path}")
    plt.close()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def save_summary_statistics(all_data, output_dir):
    """Save summary statistics"""
    
    all_results = []
    
    for subject in all_data:
        for muscle in all_data[subject]:
            data = all_data[subject][muscle]
            
            for condition in ['healthy', 'stroke']:
                df = data[condition]
                
                if len(df) == 0:
                    continue
                
                result = {
                    'subject': subject,
                    'muscle': muscle,
                    'condition': condition,
                    'n_trials': len(df),
                }
                
                for col in df.columns:
                    if col != 'trial':
                        result[f'{col}_mean'] = df[col].mean()
                        result[f'{col}_std'] = df[col].std()
                
                all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / 'morphology_summary_statistics.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    return results_df

def print_key_findings(all_data):
    """Print interpretable summary"""
    
    print("\n" + "="*70)
    print("  WAVEFORM MORPHOLOGY ANALYSIS SUMMARY")
    print("="*70)
    
    for subject in all_data:
        print(f"\n{subject}:")
        for muscle in all_data[subject]:
            data = all_data[subject][muscle]
            df_h = data['healthy']
            df_s = data['stroke']
            
            if len(df_h) == 0 or len(df_s) == 0:
                continue
            
            print(f"\n  {muscle}:")
            
            # Amplitude
            h_amp = df_h['peak_to_peak_amplitude'].mean()
            s_amp = df_s['peak_to_peak_amplitude'].mean()
            amp_change = ((s_amp - h_amp) / h_amp * 100) if h_amp > 0 else 0
            print(f"    Amplitude Change: {amp_change:+.1f}% ({h_amp:.1f} → {s_amp:.1f} µV)")
            
            # Dispersion
            h_dur = df_h['mep_duration'].mean()
            s_dur = df_s['mep_duration'].mean()
            dur_change = ((s_dur - h_dur) / h_dur * 100) if h_dur > 0 else 0
            print(f"    Temporal Dispersion: {dur_change:+.1f}% change")
            
            # Fragmentation
            h_poly = df_h['is_polyphasic'].sum() / len(df_h) * 100
            s_poly = df_s['is_polyphasic'].sum() / len(df_s) * 100
            print(f"    Polyphasic Responses: {h_poly:.1f}% → {s_poly:.1f}%")
            
            # Desynchronization
            h_smooth = df_h['smoothness'].mean()
            s_smooth = df_s['smoothness'].mean()
            smooth_change = ((s_smooth - h_smooth) / h_smooth * 100) if h_smooth > 0 else 0
            print(f"    Desynchronization: {smooth_change:+.1f}% change")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MEP Waveform Morphology Analysis')
    parser.add_argument('base_path', type=Path,
                       help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory')
    parser.add_argument('--fs', type=int, default=2000,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Number of trials to plot')
    parser.add_argument('--max-amplitude', type=float, default=500,
                       help='Maximum amplitude for artifact rejection (µV)')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.base_path / 'morphology_analysis'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("  MEP WAVEFORM MORPHOLOGY ANALYSIS")
    print("  WITH ARTIFACT REJECTION")
    print("="*70)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔊 Sampling rate: {args.fs} Hz")
    print(f"🚫 Max amplitude threshold: {args.max_amplitude} µV")
    
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})
    
    try:
        # Load data
        print("\n📥 Loading and processing data...")
        all_data = load_and_organize_data(args.base_path, fs=args.fs, 
                                         max_amplitude=args.max_amplitude)
        
        if not all_data:
            print("\n❌ No data loaded!")
            return
        
        # Create visualizations
        print("\n📊 Creating waveform overlays...")
        plot_waveform_overlays(all_data, args.output_dir, args.n_trials)
        
        print("\n📊 Creating morphology comparison...")
        plot_morphology_comparison(all_data, args.output_dir)
        
        print("\n📊 Creating fragmentation analysis...")
        plot_fragmentation_summary(all_data, args.output_dir)
        
        # Save statistics
        print("\n💾 Saving summary statistics...")
        save_summary_statistics(all_data, args.output_dir)
        
        # Print summary
        print_key_findings(all_data)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"\n📂 Output files in: {args.output_dir}")
        print(f"   • waveform_overlays.png")
        print(f"   • morphology_comparison.png")
        print(f"   • fragmentation_summary.png")
        print(f"   • morphology_summary_statistics.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()