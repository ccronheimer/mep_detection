#!/usr/bin/env python3
"""
Outlier-Cleaned MEP Analysis Script
Removes extreme outliers and provides clean analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import argparse

# Set clean style for publication
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

# Experiment configurations
ANALYSIS_CONFIGS = {
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "healthy_channels": [2, 4, 7],
        "healthy_names": ["Right Upper", "Right Forearm", "Right Hand"],
        "stroke_channels": [8, 9, 11],
        "stroke_names": ["Left Upper", "Left Forearm", "Left Hand"]
    },
    "Nov5_Chive": {
        "description": "Nov5 Chive Experiment (Ch2,10 noisy)",
        "healthy_channels": [9, 2, 8],
        "healthy_names": ["Right Upper", "Right Forearm", "Right Hand"],
        "stroke_channels": [4, 10, 7],
        "stroke_names": ["Left Upper", "Left Forearm", "Left Hand"]
    },
    "Nov5_Cheddar": {
        "description": "Nov5 Cheddar Experiment",
        "healthy_channels": [2, 4, 12],
        "healthy_names": ["Right Upper", "Right Forearm", "Right Hand"],
        "stroke_channels": [14, 9, 13],
        "stroke_names": ["Left Upper", "Left Forearm", "Left Hand"]
    }
}

def find_mep_results_csv(input_path: Path) -> Path:
    """Find the MEP results CSV file in the given path"""
    if input_path.is_file() and input_path.suffix == '.csv':
        return input_path
    
    if input_path.is_dir():
        possible_files = [
            input_path / 'mep_results.csv',
            input_path / 'hemisphere_aware_mep_results.csv'
        ]
        
        for csv_file in possible_files:
            if csv_file.exists():
                return csv_file
        
        csv_files = list(input_path.glob('*.csv'))
        if csv_files:
            return csv_files[0]
    
    raise FileNotFoundError(f"No MEP results CSV found in {input_path}")

def get_analysis_config(csv_path, config_name=None):
    """Auto-detect experiment configuration"""
    folder_name = csv_path.parent.name
    
    if config_name and config_name in ANALYSIS_CONFIGS:
        config = ANALYSIS_CONFIGS[config_name].copy()
        print(f"📋 Using specified config: {config_name}")
        return config
    
    if folder_name in ANALYSIS_CONFIGS:
        config = ANALYSIS_CONFIGS[folder_name].copy()
        print(f"📋 Auto-detected config: {folder_name}")
        return config
    
    for key in ANALYSIS_CONFIGS:
        if key.lower() in folder_name.lower() or folder_name.lower() in key.lower():
            config = ANALYSIS_CONFIGS[key].copy()
            print(f"📋 Matched config: {key} for folder {folder_name}")
            return config
    
    default_key = list(ANALYSIS_CONFIGS.keys())[0]
    config = ANALYSIS_CONFIGS[default_key].copy()
    print(f"⚠️ No matching config found for '{folder_name}', using default: {default_key}")
    return config

def remove_extreme_outliers(data: pd.Series, method='iqr_conservative', upper_limit=1000) -> tuple:
    """Remove extreme outliers using various methods"""
    
    if len(data) == 0:
        return data, pd.Series([], dtype=bool), {}
    
    original_count = len(data)
    
    if method == 'iqr_conservative':
        # Conservative IQR method (removes only extreme outliers)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Use 3.0 * IQR instead of 1.5 for more conservative outlier removal
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        
        # Also apply physiological upper limit for MEPs
        upper_bound = min(upper_bound, upper_limit)
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'physiological':
        # Remove based on physiological limits only
        outlier_mask = (data < 0) | (data > upper_limit)
        
    elif method == 'percentile':
        # Remove top and bottom percentiles
        lower_bound = data.quantile(0.01)  # Remove bottom 1%
        upper_bound = data.quantile(0.99)  # Remove top 1%
        upper_bound = min(upper_bound, upper_limit)
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    clean_data = data[~outlier_mask]
    removed_count = original_count - len(clean_data)
    
    removal_stats = {
        'original_count': original_count,
        'removed_count': removed_count,
        'percent_removed': (removed_count / original_count * 100) if original_count > 0 else 0,
        'method': method,
        'upper_limit': upper_limit,
        'final_range': (clean_data.min(), clean_data.max()) if len(clean_data) > 0 else (0, 0)
    }
    
    return clean_data, outlier_mask, removal_stats

def load_and_clean_mep_data(csv_path: Path, config: dict, outlier_method='iqr_conservative', upper_limit=1000) -> tuple:
    """Load MEP results and clean outliers"""
    
    print("📊 LOADING AND CLEANING MEP DATA")
    print("=" * 35)
    
    df = pd.read_csv(csv_path)
    
    # Separate by hemisphere
    healthy_data = df[df['hemisphere'] == 'healthy'].copy()
    stroke_data = df[df['hemisphere'] == 'stroke'].copy()
    
    print(f"✅ Loaded {len(healthy_data)} healthy + {len(stroke_data)} stroke MEPs")
    print(f"🧹 Cleaning outliers using {outlier_method} method (upper limit: {upper_limit}µV)")
    
    muscle_data = {}
    cleaning_report = {}
    
    for i, (healthy_ch, stroke_ch, muscle_name) in enumerate(zip(
        config['healthy_channels'], 
        config['stroke_channels'],
        ['Upper Arm', 'Forearm', 'Hand']
    )):
        healthy_col = f'ch{healthy_ch}_amplitude'
        stroke_col = f'ch{stroke_ch}_amplitude'
        
        # Get raw data
        healthy_raw = healthy_data[healthy_col].dropna() if healthy_col in healthy_data.columns else pd.Series([])
        stroke_raw = stroke_data[stroke_col].dropna() if stroke_col in stroke_data.columns else pd.Series([])
        
        # Clean outliers
        healthy_clean, healthy_outliers, healthy_stats = remove_extreme_outliers(healthy_raw, outlier_method, upper_limit)
        stroke_clean, stroke_outliers, stroke_stats = remove_extreme_outliers(stroke_raw, outlier_method, upper_limit)
        
        muscle_data[muscle_name] = {
            'healthy': healthy_clean,
            'stroke': stroke_clean,
            'healthy_raw': healthy_raw,
            'stroke_raw': stroke_raw,
            'healthy_channel': healthy_ch,
            'stroke_channel': stroke_ch,
            'healthy_name': config['healthy_names'][i],
            'stroke_name': config['stroke_names'][i]
        }
        
        cleaning_report[muscle_name] = {
            'healthy': healthy_stats,
            'stroke': stroke_stats
        }
        
        print(f"   {muscle_name}:")
        print(f"     Healthy: {healthy_stats['removed_count']}/{healthy_stats['original_count']} removed ({healthy_stats['percent_removed']:.1f}%)")
        print(f"     Stroke: {stroke_stats['removed_count']}/{stroke_stats['original_count']} removed ({stroke_stats['percent_removed']:.1f}%)")
        print(f"     Final ranges: H={healthy_stats['final_range'][0]:.1f}-{healthy_stats['final_range'][1]:.1f}µV, S={stroke_stats['final_range'][0]:.1f}-{stroke_stats['final_range'][1]:.1f}µV")
    
    return muscle_data, cleaning_report

def calculate_cleaned_statistics(muscle_data: dict) -> dict:
    """Calculate statistics on cleaned data"""
    
    print("\n📈 CALCULATING STATISTICS (CLEANED DATA)")
    print("=" * 40)
    
    stats_results = {}
    
    for muscle, data in muscle_data.items():
        healthy = data['healthy']
        stroke = data['stroke']
        
        if len(healthy) == 0 or len(stroke) == 0:
            print(f"   ⚠️ {muscle}: Insufficient data after cleaning")
            continue
        
        # Basic statistics
        healthy_stats = {
            'mean': healthy.mean(),
            'std': healthy.std(),
            'sem': healthy.std() / np.sqrt(len(healthy)),
            'median': healthy.median(),
            'q25': healthy.quantile(0.25),
            'q75': healthy.quantile(0.75),
            'min': healthy.min(),
            'max': healthy.max(),
            'n': len(healthy)
        }
        
        stroke_stats = {
            'mean': stroke.mean(),
            'std': stroke.std(),
            'sem': stroke.std() / np.sqrt(len(stroke)),
            'median': stroke.median(),
            'q25': stroke.quantile(0.25),
            'q75': stroke.quantile(0.75),
            'min': stroke.min(),
            'max': stroke.max(),
            'n': len(stroke)
        }
        
        # Statistical test
        try:
            statistic, p_value = stats.mannwhitneyu(healthy, stroke, alternative='two-sided')
        except:
            statistic, p_value = 0, 1.0
        
        # Impairment calculations
        mean_impairment = (1 - stroke_stats['mean'] / healthy_stats['mean']) * 100 if healthy_stats['mean'] > 0 else 0
        median_impairment = (1 - stroke_stats['median'] / healthy_stats['median']) * 100 if healthy_stats['median'] > 0 else 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(healthy) - 1) * np.var(healthy, ddof=1) + 
                            (len(stroke) - 1) * np.var(stroke, ddof=1)) / 
                           (len(healthy) + len(stroke) - 2))
        cohens_d = (healthy_stats['mean'] - stroke_stats['mean']) / pooled_std if pooled_std > 0 else 0
        
        stats_results[muscle] = {
            'healthy': healthy_stats,
            'stroke': stroke_stats,
            'p_value': p_value,
            'mean_impairment_percent': mean_impairment,
            'median_impairment_percent': median_impairment,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'healthy_name': data['healthy_name'],
            'stroke_name': data['stroke_name']
        }
        
        print(f"   {muscle}:")
        print(f"     Mean impairment: {mean_impairment:.1f}%")
        print(f"     Median impairment: {median_impairment:.1f}%")
        print(f"     Effect size (d): {cohens_d:.2f}")
        print(f"     p-value: {p_value:.4f}")
        print(f"     Sample sizes: H={len(healthy)}, S={len(stroke)}")
    
    return stats_results

def create_cleaned_scatter_plot(muscle_data: dict, stats_results: dict, config: dict, 
                               cleaning_report: dict, output_dir: Path):
    """Create scatter plot with cleaned data"""
    
    print("\n📊 CREATING CLEANED SCATTER PLOT")
    print("=" * 35)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    scatter_color = '#E91E63'
    muscle_positions = [1, 2, 3]
    muscle_names = ['Upper Arm', 'Forearm', 'Hand']
    
    # Calculate y-limits for clean visualization
    all_healthy_data = []
    all_stroke_data = []
    
    for muscle in muscle_names:
        if muscle in muscle_data:
            all_healthy_data.extend(muscle_data[muscle]['healthy'].tolist())
            all_stroke_data.extend(muscle_data[muscle]['stroke'].tolist())
    
    healthy_y_max = max(all_healthy_data) * 1.15 if all_healthy_data else 100
    stroke_y_max = max(all_stroke_data) * 1.15 if all_stroke_data else 100
    
    # Plot 1: Healthy Hemisphere (Cleaned)
    ax1.set_title(f'Healthy Hemisphere - CLEANED DATA\n({config["description"]})', 
                  fontsize=16, fontweight='bold', color='black', pad=20)
    
    for i, muscle in enumerate(muscle_names):
        if muscle in muscle_data and muscle in stats_results:
            healthy_data = muscle_data[muscle]['healthy']
            stats = stats_results[muscle]['healthy']
            
            if len(healthy_data) > 0:
                x_pos = muscle_positions[i]
                jitter = np.random.normal(0, 0.06, len(healthy_data))
                x_coords = np.full(len(healthy_data), x_pos) + jitter
                
                # Plot individual points
                ax1.scatter(x_coords, healthy_data, alpha=0.5, s=30, 
                           color=scatter_color, edgecolors='none')
                
                # Plot mean line
                mean_line_width = 0.15
                ax1.plot([x_pos - mean_line_width, x_pos + mean_line_width], 
                        [stats['mean'], stats['mean']], 
                        color='black', linewidth=4, solid_capstyle='round', zorder=10)
                
                # Add sample size and cleaning info
                removed_pct = cleaning_report[muscle]['healthy']['percent_removed']
                ax1.text(x_pos, -healthy_y_max*0.08, f'n={stats["n"]}\n({removed_pct:.0f}% removed)', 
                        ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(0, healthy_y_max)
    ax1.set_xticks(muscle_positions)
    ax1.set_xticklabels(muscle_names, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Muscle Groups', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MEP Amplitude (µV)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stroke Hemisphere (Cleaned)
    ax2.set_title(f'Stroke Hemisphere - CLEANED DATA\n({config["description"]})', 
                  fontsize=16, fontweight='bold', color='black', pad=20)
    
    for i, muscle in enumerate(muscle_names):
        if muscle in muscle_data and muscle in stats_results:
            stroke_data = muscle_data[muscle]['stroke']
            stats = stats_results[muscle]['stroke']
            
            if len(stroke_data) > 0:
                x_pos = muscle_positions[i]
                jitter = np.random.normal(0, 0.06, len(stroke_data))
                x_coords = np.full(len(stroke_data), x_pos) + jitter
                
                # Plot individual points
                ax2.scatter(x_coords, stroke_data, alpha=0.5, s=30, 
                           color=scatter_color, edgecolors='none')
                
                # Plot mean line
                mean_line_width = 0.15
                ax2.plot([x_pos - mean_line_width, x_pos + mean_line_width], 
                        [stats['mean'], stats['mean']], 
                        color='black', linewidth=4, solid_capstyle='round', zorder=10)
                
                # Add sample size and cleaning info
                removed_pct = cleaning_report[muscle]['stroke']['percent_removed']
                ax2.text(x_pos, -stroke_y_max*0.08, f'n={stats["n"]}\n({removed_pct:.0f}% removed)', 
                        ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(0.5, 3.5)
    ax2.set_ylim(0, stroke_y_max)
    ax2.set_xticks(muscle_positions)
    ax2.set_xticklabels(muscle_names, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Muscle Groups', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MEP Amplitude (µV)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_color, 
                  markersize=8, alpha=0.6, label='Individual MEPs (cleaned)'),
        plt.Line2D([0], [0], color='black', linewidth=4, 
                  label='Mean amplitude', solid_capstyle='round')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               fontsize=12, bbox_to_anchor=(0.5, -0.02))
    
    # Add significance stars
    for i, muscle in enumerate(muscle_names):
        if muscle in stats_results and stats_results[muscle]['significant']:
            # Put star on the hemisphere with higher mean
            healthy_mean = stats_results[muscle]['healthy']['mean']
            stroke_mean = stats_results[muscle]['stroke']['mean']
            
            if healthy_mean > stroke_mean:
                star_y = healthy_mean + healthy_y_max * 0.05
                ax1.text(muscle_positions[i], star_y, '***', ha='center', va='bottom', 
                        fontsize=20, fontweight='bold', color='red')
            else:
                star_y = stroke_mean + stroke_y_max * 0.05
                ax2.text(muscle_positions[i], star_y, '***', ha='center', va='bottom', 
                        fontsize=20, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    plot_path = output_dir / f'{config["description"].replace(" ", "_")}_CLEANED_mep_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Cleaned scatter plot saved: {plot_path}")
    return fig

def export_cleaning_report(cleaning_report: dict, stats_results: dict, config: dict, output_dir: Path):
    """Export detailed cleaning and analysis report"""
    
    print("\n📋 EXPORTING CLEANING REPORT")
    print("=" * 30)
    
    # Create comprehensive report
    report_path = output_dir / f'{config["description"].replace(" ", "_")}_CLEANING_REPORT.txt'
    
    with open(report_path, 'w') as f:
        f.write("OUTLIER CLEANING AND MEP ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {config['description']}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        f.write("OUTLIER REMOVAL SUMMARY:\n")
        f.write("-" * 25 + "\n")
        
        total_removed = 0
        total_original = 0
        
        for muscle, report in cleaning_report.items():
            f.write(f"\n{muscle.upper()}:\n")
            
            h_removed = report['healthy']['removed_count']
            h_original = report['healthy']['original_count']
            s_removed = report['stroke']['removed_count']
            s_original = report['stroke']['original_count']
            
            total_removed += h_removed + s_removed
            total_original += h_original + s_original
            
            f.write(f"  Healthy: {h_removed}/{h_original} removed ({report['healthy']['percent_removed']:.1f}%)\n")
            f.write(f"  Stroke: {s_removed}/{s_original} removed ({report['stroke']['percent_removed']:.1f}%)\n")
            f.write(f"  Method: {report['healthy']['method']}\n")
            f.write(f"  Upper limit: {report['healthy']['upper_limit']}µV\n")
        
        f.write(f"\nOVERALL CLEANING:\n")
        f.write(f"  Total removed: {total_removed}/{total_original} ({total_removed/total_original*100:.1f}%)\n")
        
        f.write(f"\nCLEANED DATA ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        
        for muscle, result in stats_results.items():
            f.write(f"\n{muscle.upper()}:\n")
            f.write(f"  Healthy: {result['healthy']['mean']:.1f} ± {result['healthy']['sem']:.1f}µV (n={result['healthy']['n']})\n")
            f.write(f"  Stroke: {result['stroke']['mean']:.1f} ± {result['stroke']['sem']:.1f}µV (n={result['stroke']['n']})\n")
            f.write(f"  Mean impairment: {result['mean_impairment_percent']:.1f}%\n")
            f.write(f"  Median impairment: {result['median_impairment_percent']:.1f}%\n")
            f.write(f"  Effect size (Cohen's d): {result['cohens_d']:.2f}\n")
            f.write(f"  P-value: {result['p_value']:.6f}\n")
            f.write(f"  Significant: {'YES' if result['significant'] else 'NO'}\n")
        
        f.write(f"\nRECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        
        if total_removed / total_original > 0.20:
            f.write("• HIGH REMOVAL RATE: >20% of data removed - consider investigating detection parameters\n")
        elif total_removed / total_original > 0.10:
            f.write("• MODERATE REMOVAL RATE: 10-20% of data removed - acceptable for artifact removal\n")
        else:
            f.write("• LOW REMOVAL RATE: <10% of data removed - good data quality\n")
        
        f.write("• Use cleaned data for publication\n")
        f.write("• Report outlier removal methodology in methods section\n")
        f.write("• Consider median-based statistics for additional robustness\n")
    
    print(f"✅ Cleaning report saved: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Outlier-Cleaned MEP Analysis')
    parser.add_argument('input_path', type=Path, help='MEP results CSV file or experiment folder')
    parser.add_argument('--config', type=str, help='Experiment config name (optional)')
    parser.add_argument('--output-dir', type=Path, help='Output directory (optional)')
    parser.add_argument('--upper-limit', type=int, default=1000, help='Upper limit for MEP values (µV)')
    parser.add_argument('--method', type=str, default='iqr_conservative', 
                       choices=['iqr_conservative', 'physiological', 'percentile'],
                       help='Outlier removal method')
    
    args = parser.parse_args()
    
    # Find the CSV file
    try:
        csv_file = find_mep_results_csv(args.input_path)
        print(f"📄 Found MEP results: {csv_file.name}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Set output directory
    if not args.output_dir:
        args.output_dir = csv_file.parent
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("    OUTLIER-CLEANED MEP ANALYSIS")
    print("="*60)
    print(f"📁 Input: {csv_file}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🧹 Cleaning method: {args.method}")
    print(f"📏 Upper limit: {args.upper_limit}µV")
    
    try:
        # Get configuration
        config = get_analysis_config(csv_file, args.config)
        
        # Load and clean data
        muscle_data, cleaning_report = load_and_clean_mep_data(
            csv_file, config, args.method, args.upper_limit
        )
        
        # Calculate statistics on cleaned data
        stats_results = calculate_cleaned_statistics(muscle_data)
        
        if not stats_results:
            print("❌ No valid muscle data found after cleaning")
            return
        
        # Create cleaned scatter plot
        scatter_fig = create_cleaned_scatter_plot(
            muscle_data, stats_results, config, cleaning_report, args.output_dir
        )
        
        # Export cleaning report
        report_path = export_cleaning_report(
            cleaning_report, stats_results, config, args.output_dir
        )
        
        # Print final summary
        print(f"\n" + "="*60)
        print("           CLEANED ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\n🎯 KEY FINDINGS (CLEANED DATA):")
        for muscle, result in stats_results.items():
            sig_text = "***" if result['significant'] else "ns"
            effect_size = abs(result['cohens_d'])
            effect_desc = "large" if effect_size > 0.8 else "medium" if effect_size > 0.5 else "small"
            print(f"   {muscle}: {result['mean_impairment_percent']:.1f}% impairment, d={result['cohens_d']:.2f} ({effect_desc}), {sig_text}")
        
        # Calculate total data removed
        total_removed = sum(
            cleaning_report[muscle]['healthy']['removed_count'] + 
            cleaning_report[muscle]['stroke']['removed_count']
            for muscle in cleaning_report
        )
        total_original = sum(
            cleaning_report[muscle]['healthy']['original_count'] + 
            cleaning_report[muscle]['stroke']['original_count']
            for muscle in cleaning_report
        )
        
        removal_pct = total_removed / total_original * 100 if total_original > 0 else 0
        
        print(f"\n🧹 CLEANING SUMMARY:")
        print(f"   Total outliers removed: {total_removed}/{total_original} ({removal_pct:.1f}%)")
        print(f"   Upper limit applied: {args.upper_limit}µV")
        print(f"   Method used: {args.method}")
        
        if removal_pct > 20:
            print(f"   ⚠️ High removal rate - consider reviewing detection parameters")
        elif removal_pct > 10:
            print(f"   ✅ Moderate removal rate - acceptable for artifact cleaning")
        else:
            print(f"   ✅ Low removal rate - good original data quality")
        
        print(f"\n📊 FILES CREATED:")
        print(f"   Cleaned scatter plot: {config['description'].replace(' ', '_')}_CLEANED_mep_analysis.png")
        print(f"   Cleaning report: {config['description'].replace(' ', '_')}_CLEANING_REPORT.txt")
        
        print(f"\n✅ CLEANED ANALYSIS READY FOR PUBLICATION!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()