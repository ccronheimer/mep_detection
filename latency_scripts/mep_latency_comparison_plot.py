#!/usr/bin/env python3
"""
MEP Latency Subject-Focused Paired Comparison Plot
Each panel shows one subject's response across all three muscles
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

ANALYSIS_CONFIGS = {
    "NHP1": {
        "data_path": "Nov5_Olive/mep_results_with_latency.csv",
        "color": "#FF99B4",  # Pink
        "healthy_channels": [2, 4, 7],
        "stroke_channels": [8, 9, 11],
        "short_name": "NHP1",
        "show": True
    },
    "NHP2": {
        "data_path": "Nov5_Cheddar/mep_results_with_latency.csv",
        "color": "#FFB3C6",  # Light pink
        "healthy_channels": [2, 4, 12],
        "stroke_channels": [14, 9, 13],
        "short_name": "NHP2",
        "show": True
    },
    "NHP3": {
        "data_path": "Oct31_Chive/mep_results_with_latency.csv",
        "color": "#7DD3C0",  # Cyan
        "healthy_channels": [1, 2, 3],
        "stroke_channels": [4, 5, 6],
        "short_name": "NHP3",
        "show": True
    }
}

MUSCLE_MAPPING = {
    0: "Biceps",
    1: "Brach", 
    2: "APB"
}

PLOT_SETTINGS = {
    "figure_size": (16, 6),
    "dpi": 300,
    "point_size": 80,
    "line_width": 2,
    "alpha": 0.7
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_latency_data(base_path: Path) -> dict:
    """Load latency data for all experiments"""
    all_experiments = {}
    
    for exp_name, config in ANALYSIS_CONFIGS.items():
        if not config.get('show', True):
            continue
            
        file_path = base_path / config['data_path']
        
        if not file_path.exists():
            print(f"⚠️  Skipping {exp_name}: File not found at {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            if 'latency' not in df.columns:
                print(f"⚠️  Skipping {exp_name}: No 'latency' column found")
                print(f"    Available columns: {list(df.columns)}")
                continue
            
            # Organize data by muscle
            muscle_data = {}
            for muscle_idx, muscle_name in MUSCLE_MAPPING.items():
                # Healthy hemisphere
                healthy_channel = config['healthy_channels'][muscle_idx]
                healthy_data = df[df['channel'] == healthy_channel]['latency'].dropna()
                
                # Stroke hemisphere
                stroke_channel = config['stroke_channels'][muscle_idx]
                stroke_data = df[df['channel'] == stroke_channel]['latency'].dropna()
                
                muscle_data[muscle_name] = {
                    'healthy': healthy_data,
                    'stroke': stroke_data
                }
            
            all_experiments[exp_name] = {
                'config': config,
                'muscle_data': muscle_data
            }
            
            print(f"✓ Loaded {exp_name}: {len(df)} total MEPs")
            
        except Exception as e:
            print(f"❌ Error loading {exp_name}: {e}")
            continue
    
    return all_experiments

# =============================================================================
# SUBJECT-FOCUSED PAIRED COMPARISON PLOT
# =============================================================================

def create_subject_focused_plot(all_experiments: dict, output_dir: Path):
    """Create subject-focused plot showing all muscles for each subject"""
    
    print("\n📊 CREATING SUBJECT-FOCUSED PAIRED COMPARISON PLOT")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 3, figsize=PLOT_SETTINGS["figure_size"], sharey=True)
    fig.suptitle('MEP Latencies in Healthy vs. Stroke-Affected Hemispheres', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    all_statistics = []
    
    # Iterate through subjects (one panel per subject)
    for subject_idx, (ax, (exp_name, exp_data)) in enumerate(zip(axes, all_experiments.items())):
        
        muscle_names = list(MUSCLE_MAPPING.values())
        muscle_data_list = []
        
        # Collect data for all muscles in this subject
        for muscle_name in muscle_names:
            if muscle_name in exp_data['muscle_data']:
                healthy = exp_data['muscle_data'][muscle_name]['healthy']
                stroke = exp_data['muscle_data'][muscle_name]['stroke']
                
                # Calculate means and SEM
                healthy_mean = healthy.mean() if len(healthy) > 0 else np.nan
                stroke_mean = stroke.mean() if len(stroke) > 0 else np.nan
                healthy_sem = healthy.std() / np.sqrt(len(healthy)) if len(healthy) > 0 else 0
                stroke_sem = stroke.std() / np.sqrt(len(stroke)) if len(stroke) > 0 else 0
                
                muscle_data_list.append({
                    'muscle': muscle_name,
                    'healthy_mean': healthy_mean,
                    'stroke_mean': stroke_mean,
                    'healthy_sem': healthy_sem,
                    'stroke_sem': stroke_sem,
                    'healthy_all': healthy,
                    'stroke_all': stroke
                })
        
        # Plot data for this subject
        x_positions = np.arange(len(muscle_data_list))
        
        for idx, data in enumerate(muscle_data_list):
            # Plot connecting line
            if not (np.isnan(data['healthy_mean']) or np.isnan(data['stroke_mean'])):
                ax.plot([idx-0.15, idx+0.15], 
                       [data['healthy_mean'], data['stroke_mean']],
                       color=exp_data['config']['color'], 
                       linewidth=PLOT_SETTINGS['line_width'],
                       alpha=PLOT_SETTINGS['alpha'],
                       zorder=1)
            
            # Plot healthy point with error bar
            if not np.isnan(data['healthy_mean']):
                ax.errorbar(idx-0.15, data['healthy_mean'],
                           yerr=data['healthy_sem'],
                           fmt='o',
                           markersize=8,
                           color=exp_data['config']['color'],
                           markeredgecolor='white',
                           markeredgewidth=2,
                           ecolor=exp_data['config']['color'],
                           elinewidth=1.5,
                           capsize=4,
                           capthick=1.5,
                           label='Healthy' if idx == 0 else '',
                           zorder=3)
            
            # Plot stroke point with error bar
            if not np.isnan(data['stroke_mean']):
                ax.errorbar(idx+0.15, data['stroke_mean'],
                           yerr=data['stroke_sem'],
                           fmt='s',
                           markersize=8,
                           color=exp_data['config']['color'],
                           markeredgecolor='white',
                           markeredgewidth=2,
                           ecolor=exp_data['config']['color'],
                           elinewidth=1.5,
                           capsize=4,
                           capthick=1.5,
                           label='Stroke' if idx == 0 else '',
                           zorder=3)
        
        # Styling
        ax.set_title(f'{exp_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Muscle', fontsize=12, fontweight='bold')
        if subject_idx == 0:
            ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        
        # Set y-axis
        max_y = max([d['healthy_mean'] + d['healthy_sem'] 
                    if not np.isnan(d['healthy_mean']) else 0 for d in muscle_data_list] + 
                   [d['stroke_mean'] + d['stroke_sem'] 
                    if not np.isnan(d['stroke_mean']) else 0 for d in muscle_data_list])
        ax.set_ylim(0.0, max_y * 1.15)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([d['muscle'] for d in muscle_data_list], rotation=0)
        ax.grid(False)  # No grid lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend only to first subplot
        if subject_idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='gray', markersize=10, 
                          markeredgecolor='white', markeredgewidth=2,
                          label='Healthy'),
                plt.Line2D([0], [0], marker='s', color='w', 
                          markerfacecolor='gray', markersize=10,
                          markeredgecolor='white', markeredgewidth=2,
                          label='Stroke')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     frameon=True, fontsize=10)
        
        # Statistical analysis and add significance markers
        for idx, data in enumerate(muscle_data_list):
            healthy_vals = data['healthy_all']
            stroke_vals = data['stroke_all']
            
            if len(healthy_vals) > 0 and len(stroke_vals) > 0:
                # Independent samples t-test
                t_stat, p_value = stats.ttest_ind(healthy_vals, stroke_vals)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((healthy_vals.std()**2 + stroke_vals.std()**2) / 2)
                cohens_d = (data['stroke_mean'] - data['healthy_mean']) / pooled_std if pooled_std > 0 else np.nan
                
                # Add significance marker only if significant
                if not (np.isnan(data['healthy_mean']) or np.isnan(data['stroke_mean'])):
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = None
                    
                    if sig_text:
                        y_pos = max(data['healthy_mean'] + data['healthy_sem'], 
                                   data['stroke_mean'] + data['stroke_sem']) * 1.05
                        x_pos = idx
                        
                        ax.text(x_pos, y_pos, sig_text, 
                               ha='center', va='bottom', 
                               fontsize=10, fontweight='bold',
                               color='black')
                
                all_statistics.append({
                    'Subject': exp_name,
                    'Muscle': data['muscle'],
                    'Healthy_Mean_ms': data['healthy_mean'],
                    'Healthy_SEM_ms': data['healthy_sem'],
                    'Stroke_Mean_ms': data['stroke_mean'],
                    'Stroke_SEM_ms': data['stroke_sem'],
                    'Difference_ms': data['stroke_mean'] - data['healthy_mean'],
                    'Percent_Change': ((data['stroke_mean'] - data['healthy_mean']) / data['healthy_mean'] * 100),
                    'N_Healthy': len(healthy_vals),
                    'N_Stroke': len(stroke_vals),
                    'P_Value': p_value,
                    'Cohens_D': cohens_d
                })
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot
    plot_path = output_dir / 'mep_latency_subject_focused.png'
    plt.savefig(plot_path, dpi=PLOT_SETTINGS['dpi'], 
               bbox_inches='tight', facecolor='white')
    print(f"✓ Saved plot: {plot_path}")
    
    # Save statistics
    stats_df = pd.DataFrame(all_statistics)
    stats_path = output_dir / 'latency_subject_focused_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved statistics: {stats_path}")
    
    return fig, stats_df

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(stats_df: pd.DataFrame):
    """Print summary of statistical findings"""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY - SUBJECT-FOCUSED")
    print("="*70)
    print("Statistical test: Independent samples t-test")
    print("Compares distributions of individual MEP trials (n=500-1100 per condition)")
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print("Error bars represent SEM (standard error of the mean)")
    
    for subject in stats_df['Subject'].unique():
        subject_data = stats_df[stats_df['Subject'] == subject]
        
        print(f"\n{subject}:")
        print("-" * 50)
        
        for _, row in subject_data.iterrows():
            significance = "***" if row['P_Value'] < 0.001 else \
                          "**" if row['P_Value'] < 0.01 else \
                          "*" if row['P_Value'] < 0.05 else "ns"
            
            print(f"  {row['Muscle']}:")
            print(f"    Healthy: {row['Healthy_Mean_ms']:.2f} ± {row['Healthy_SEM_ms']:.2f} ms (n={row['N_Healthy']:.0f})")
            print(f"    Stroke:  {row['Stroke_Mean_ms']:.2f} ± {row['Stroke_SEM_ms']:.2f} ms (n={row['N_Stroke']:.0f})")
            print(f"    Change:  {row['Difference_ms']:+.2f} ms ({row['Percent_Change']:+.1f}%) {significance}")
            print(f"    p-value: {row['P_Value']:.4f}, Cohen's d = {row['Cohens_D']:.2f}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create subject-focused MEP latency comparison plot')
    parser.add_argument('base_path', type=Path, 
                       help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: base_path/latency_paired_results)')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.base_path / 'latency_paired_results'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("  MEP LATENCY SUBJECT-FOCUSED PAIRED COMPARISON")
    print("="*70)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"📊 Statistical test: Independent samples t-test")
    print(f"📊 Layout: One panel per subject, all muscles within each panel")
    
    # Configure matplotlib
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    
    try:
        # Load data
        all_experiments = load_latency_data(args.base_path)
        
        if not all_experiments:
            print("\n❌ No latency data found!")
            print("\n💡 Make sure you've run the latency detection script first:")
            print("   python mep_latency_detector.py <experiment_folder>/")
            return
        
        print(f"\n✅ Successfully loaded {len(all_experiments)} experiments")
        
        # Create subject-focused plot
        fig, all_stats = create_subject_focused_plot(all_experiments, args.output_dir)
        
        # Print summary
        print_summary_statistics(all_stats)
        
        print(f"\n✅ SUBJECT-FOCUSED ANALYSIS COMPLETE!")
        print(f"\n📂 Output files in: {args.output_dir}")
        print(f"   • mep_latency_subject_focused.png (main figure)")
        print(f"   • latency_subject_focused_statistics.csv (detailed stats)")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()