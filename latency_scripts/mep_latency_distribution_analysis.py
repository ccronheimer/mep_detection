#!/usr/bin/env python3
"""
MEP Latency Distribution and Bimodality Analysis
Analyzes latency distributions to detect pathway switching and dual-pathway activation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

ANALYSIS_CONFIGS = {
    "NHP1": {
        "data_path": "Nov5_Olive/mep_results_with_latency.csv",
        "color": "#FF99B4",
        "healthy_channels": [2, 4, 7],
        "stroke_channels": [8, 9, 11],
    },
    "NHP2": {
        "data_path": "Nov5_Cheddar/mep_results_with_latency.csv",
        "color": "#FFB3C6",
        "healthy_channels": [2, 4, 12],
        "stroke_channels": [14, 9, 13],
    },
    "NHP3": {
        "data_path": "Oct31_Chive/mep_results_with_latency.csv",
        "color": "#7DD3C0",
        "healthy_channels": [1, 2, 3],
        "stroke_channels": [4, 5, 6],
    }
}

MUSCLE_MAPPING = {0: "Biceps", 1: "Brach", 2: "APB"}

# =============================================================================
# BIMODALITY TESTS
# =============================================================================

def bimodality_coefficient(data):
    """
    Calculate bimodality coefficient (BC)
    BC > 0.555 suggests bimodality
    """
    n = len(data)
    if n < 3:
        return np.nan
    
    m3 = stats.skew(data)
    m4 = stats.kurtosis(data, fisher=False)  # Use Pearson's definition
    
    numerator = m3**2 + 1
    denominator = m4 + (3 * (n-1)**2) / ((n-2)*(n-3))
    
    bc = numerator / denominator
    return bc

def hartigan_dip_test(data):
    """
    Hartigan's Dip Test for unimodality
    Returns p-value (low p = reject unimodality = bimodal)
    Note: Requires diptest package (pip install diptest)
    """
    try:
        from diptest import diptest
        dip_stat, p_value = diptest(data)
        return dip_stat, p_value
    except ImportError:
        print("⚠️  diptest not installed. Install with: pip install diptest")
        return None, None

def fit_gaussian_mixture(data, n_components=2):
    """
    Fit Gaussian Mixture Model to test for multiple modes
    """
    from sklearn.mixture import GaussianMixture
    
    if len(data) < 10:
        return None, None
    
    data_reshaped = data.values.reshape(-1, 1)
    
    # Fit models with 1 and 2 components
    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm2 = GaussianMixture(n_components=n_components, random_state=42)
    
    gmm1.fit(data_reshaped)
    gmm2.fit(data_reshaped)
    
    # Compare using BIC (lower is better)
    bic1 = gmm1.bic(data_reshaped)
    bic2 = gmm2.bic(data_reshaped)
    
    # If BIC improvement > 10, prefer 2-component model
    bic_improvement = bic1 - bic2
    
    return gmm2, bic_improvement

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_organize_data(base_path: Path):
    """Load latency data organized by subject, muscle, hemisphere"""
    all_data = {}
    
    for subject, config in ANALYSIS_CONFIGS.items():
        file_path = base_path / config['data_path']
        
        if not file_path.exists():
            print(f"⚠️  Skipping {subject}: File not found")
            continue
        
        df = pd.read_csv(file_path)
        
        if 'latency' not in df.columns:
            print(f"⚠️  Skipping {subject}: No latency column")
            continue
        
        subject_data = {}
        
        for muscle_idx, muscle_name in MUSCLE_MAPPING.items():
            healthy_ch = config['healthy_channels'][muscle_idx]
            stroke_ch = config['stroke_channels'][muscle_idx]
            
            healthy_latencies = df[df['channel'] == healthy_ch]['latency'].dropna()
            stroke_latencies = df[df['channel'] == stroke_ch]['latency'].dropna()
            
            subject_data[muscle_name] = {
                'healthy': healthy_latencies,
                'stroke': stroke_latencies,
                'color': config['color']
            }
        
        all_data[subject] = subject_data
        print(f"✓ Loaded {subject}")
    
    return all_data

# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distribution(data, label):
    """Comprehensive distribution analysis"""
    results = {
        'label': label,
        'n': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'median': data.median(),
        'cv': data.std() / data.mean() if data.mean() > 0 else np.nan,
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
    }
    
    # Bimodality coefficient
    results['bimodality_coef'] = bimodality_coefficient(data)
    results['is_bimodal_bc'] = results['bimodality_coef'] > 0.555
    
    # Hartigan's dip test
    dip_stat, dip_p = hartigan_dip_test(data)
    results['dip_statistic'] = dip_stat
    results['dip_pvalue'] = dip_p
    results['is_bimodal_dip'] = dip_p < 0.05 if dip_p is not None else None
    
    # Gaussian mixture model
    gmm, bic_improvement = fit_gaussian_mixture(data)
    results['bic_improvement'] = bic_improvement
    results['is_bimodal_gmm'] = bic_improvement > 10 if bic_improvement is not None else None
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_latency_distributions(all_data, output_dir):
    """Create comprehensive latency distribution plots"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('MEP Latency Distributions', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    all_results = []
    
    subjects = list(all_data.keys())
    muscles = list(MUSCLE_MAPPING.values())
    
    # Use consistent condition-based colors
    stroke_color = '#FF6B6B'  # Red for all stroke conditions
    
    for row, subject in enumerate(subjects):
        for col, muscle in enumerate(muscles):
            ax = axes[row, col]
            
            if subject not in all_data or muscle not in all_data[subject]:
                ax.axis('off')
                continue
            
            data = all_data[subject][muscle]
            healthy = data['healthy']
            stroke = data['stroke']
            
            # Plot histograms
            bins = np.linspace(
                min(healthy.min(), stroke.min()),
                max(healthy.max(), stroke.max()),
                40
            )
            
            ax.hist(healthy, bins=bins, alpha=0.3, color='green', 
                   label='Healthy', density=True, edgecolor='darkgreen')
            ax.hist(stroke, bins=bins, alpha=0.3, color=stroke_color,
                   label='Stroke', density=True, edgecolor='darkred')
            
            # Add KDE curves
            if len(healthy) > 5:
                kde_healthy = gaussian_kde(healthy)
                x_range = np.linspace(healthy.min(), healthy.max(), 200)
                ax.plot(x_range, kde_healthy(x_range), 'g-', linewidth=2)
            
            if len(stroke) > 5:
                kde_stroke = gaussian_kde(stroke)
                x_range = np.linspace(stroke.min(), stroke.max(), 200)
                ax.plot(x_range, kde_stroke(x_range), color=stroke_color, linewidth=2)
            
            # Analyze distributions (for CSV output, not displayed)
            healthy_stats = analyze_distribution(healthy, f'{subject}_{muscle}_Healthy')
            stroke_stats = analyze_distribution(stroke, f'{subject}_{muscle}_Stroke')
            
            all_results.append(healthy_stats)
            all_results.append(stroke_stats)
            
            # Styling
            ax.set_title(f'{subject} - {muscle}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Latency (ms)', fontsize=10)
            if col == 0:
                ax.set_ylabel('Density', fontsize=10)
            
            # Move legend to upper right
            ax.legend(loc='upper right', fontsize=9, frameon=True)
            
            # Remove grid lines
            ax.grid(False)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Save
    plot_path = output_dir / 'latency_distributions_bimodality.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {plot_path}")
    
    # Save statistics
    results_df = pd.DataFrame(all_results)
    stats_path = output_dir / 'distribution_statistics.csv'
    results_df.to_csv(stats_path, index=False)
    print(f"✓ Saved: {stats_path}")
    
    return results_df

def plot_overlay_comparison(all_data, output_dir):
    """Create overlay plots for direct healthy vs stroke comparison"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('MEP Latency: Healthy vs Stroke Overlay', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    subjects = list(all_data.keys())
    muscles = list(MUSCLE_MAPPING.values())
    
    # Use consistent condition-based colors
    stroke_color = '#FF6B6B'  # Red for all stroke conditions
    
    for row, subject in enumerate(subjects):
        for col, muscle in enumerate(muscles):
            ax = axes[row, col]
            
            if subject not in all_data or muscle not in all_data[subject]:
                ax.axis('off')
                continue
            
            data = all_data[subject][muscle]
            healthy = data['healthy']
            stroke = data['stroke']
            
            # Create violin plots
            parts = ax.violinplot([healthy, stroke], 
                                  positions=[1, 2],
                                  showmeans=True,
                                  showmedians=True)
            
            # Color the violins - condition-based colors
            parts['bodies'][0].set_facecolor('green')
            parts['bodies'][0].set_alpha(0.3)
            parts['bodies'][1].set_facecolor(stroke_color)
            parts['bodies'][1].set_alpha(0.3)
            
            # Add individual points - condition-based colors
            np.random.seed(42)
            x1 = np.random.normal(1, 0.04, len(healthy))
            x2 = np.random.normal(2, 0.04, len(stroke))
            
            ax.scatter(x1, healthy, alpha=0.1, s=2, color='green')
            ax.scatter(x2, stroke, alpha=0.1, s=2, color=stroke_color)
            
            # Styling
            ax.set_title(f'{subject} - {muscle}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latency (ms)', fontsize=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Healthy', 'Stroke'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
            
            # Add statistics
            mean_healthy = healthy.mean()
            mean_stroke = stroke.mean()
            change_pct = ((mean_stroke - mean_healthy) / mean_healthy) * 100
            
            stats_text = f"Δ = {change_pct:+.1f}%"
            ax.text(0.5, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    plot_path = output_dir / 'latency_overlay_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {plot_path}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_bimodality_summary(results_df):
    """Print summary of bimodality findings"""
    
    print("\n" + "="*70)
    print("BIMODALITY ANALYSIS SUMMARY")
    print("="*70)
    print("\nBimodality Coefficient (BC) interpretation:")
    print("  BC > 0.555: Suggests bimodality (multiple pathways)")
    print("  BC < 0.555: Suggests unimodality (single pathway)")
    print("\nCoefficient of Variation (CV) interpretation:")
    print("  CV > 0.3: High variability (unstable recruitment)")
    print("  CV < 0.15: Low variability (consistent recruitment)")
    
    # Focus on stroke hemisphere
    stroke_data = results_df[results_df['label'].str.contains('Stroke')]
    
    print("\n" + "-"*70)
    print("STROKE HEMISPHERE FINDINGS:")
    print("-"*70)
    
    for _, row in stroke_data.iterrows():
        subject_muscle = row['label'].replace('_Stroke', '')
        print(f"\n{subject_muscle}:")
        print(f"  Mean latency: {row['mean']:.2f} ms")
        print(f"  CV: {row['cv']:.3f} {'(HIGH VARIABILITY)' if row['cv'] > 0.3 else '(stable)'}")
        print(f"  Bimodality Coef: {row['bimodality_coef']:.3f} ", end='')
        
        if row['is_bimodal_bc']:
            print("→ BIMODAL - Evidence of dual pathways")
        else:
            print("→ Unimodal (single pathway)")
        
        if row['dip_pvalue'] is not None:
            print(f"  Dip test p-value: {row['dip_pvalue']:.4f} ", end='')
            if row['is_bimodal_dip']:
                print("→ Significant bimodality")
            else:
                print("→ Unimodal")
    
    # Highlight key findings
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    bimodal_cases = stroke_data[stroke_data['is_bimodal_bc'] == True]
    if len(bimodal_cases) > 0:
        print("\n🔍 BIMODAL DISTRIBUTIONS DETECTED:")
        for _, row in bimodal_cases.iterrows():
            print(f"  • {row['label']}: BC={row['bimodality_coef']:.3f}")
        print("\n  ➜ Suggests dual-pathway activation (direct + indirect)")
    else:
        print("\n✓ No bimodal distributions detected")
        print("  ➜ Suggests uniform pathway damage or compensation")
    
    high_cv = stroke_data[stroke_data['cv'] > 0.3]
    if len(high_cv) > 0:
        print("\n⚠️  HIGH VARIABILITY DETECTED:")
        for _, row in high_cv.iterrows():
            print(f"  • {row['label']}: CV={row['cv']:.3f}")
        print("\n  ➜ Suggests unstable/inconsistent pathway recruitment")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze MEP latency distributions for bimodality'
    )
    parser.add_argument('base_path', type=Path,
                       help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.base_path / 'latency_distribution_analysis'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("  MEP LATENCY DISTRIBUTION & BIMODALITY ANALYSIS")
    print("="*70)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {args.output_dir}")
    
    # Configure plotting
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
    })
    
    try:
        # Load data
        print("\n📥 Loading data...")
        all_data = load_and_organize_data(args.base_path)
        
        if not all_data:
            print("\n❌ No data loaded!")
            return
        
        # Create visualizations
        print("\n📊 Creating distribution plots...")
        results_df = plot_latency_distributions(all_data, args.output_dir)
        
        print("\n📊 Creating overlay comparison...")
        plot_overlay_comparison(all_data, args.output_dir)
        
        # Print summary
        print_bimodality_summary(results_df)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"\n📂 Output files in: {args.output_dir}")
        print(f"   • latency_distributions_bimodality.png")
        print(f"   • latency_overlay_comparison.png")
        print(f"   • distribution_statistics.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()