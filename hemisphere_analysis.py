#!/usr/bin/env python3
"""
MEP Healthy vs Stroke-Affected Hemisphere Comparison
FIXED: Proper spacing, margins, and layout
Multiple layout options available
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

MONKEY_CONFIG = {
    "Nov5_Olive": {
        "show": True,
        "name": "NHP1",
        "color": "#FF69B4"
    },
    "Nov5_Cheddar": {
        "show": True,
        "name": "NHP2",
        "color": "#C71585"
    },
    "Oct31_Chive": {
        "show": True,
        "name": "NHP3",
        "color": "#FF69B4"
    }
}

PLOT_SETTINGS = {
    "figure_width": 32,
    "figure_height": 14,
    "title_font_size": 42,
    "axis_label_font_size": 36,
    "tick_label_font_size": 28,
    "muscle_group_spacing": 5,
    "y_tick_interval": 20,
    "spine_width": 3,
    "dpi": 300,
    "remove_outliers": True,
    "outlier_method": "iqr_conservative",  # MATCHES analysis script
    "outlier_upper_limit": 1000,  # MATCHES analysis script default
    "use_separate_y_scales": False,
}

MUSCLE_CONFIG = {
    "Bicep": True,
    "Brachioradialis": True,
    "Abductor Pollicis Brevis": True
}

MUSCLE_COLORS = {
    "Bicep": "#FFB3BA",
    "Brachioradialis": "#FF7A92",
    "Abductor Pollicis Brevis": "#7BC8D9"
}

ANALYSIS_CONFIGS = {
    "Nov5_Olive": {
        "short_name": "NHP1",
        "show": True,
        "data_path": "Nov5_Olive/mep_results.csv",
        "healthy": {"channels": [2, 4, 7]},
        "stroke": {"channels": [8, 9, 11]}
    },
    "Nov5_Cheddar": {
        "short_name": "NHP2",
        "show": True,
        "data_path": "Nov5_Cheddar/mep_results.csv",
        "healthy": {"channels": [2, 4, 12]},
        "stroke": {"channels": [14, 9, 13]}
    },
    "Oct31_Chive": {
        "short_name": "NHP3",
        "show": True,
        "data_path": "Oct31_Chive/mep_results.csv",
        "healthy": {"channels": [1, 2, 3]},
        "stroke": {"channels": [4, 5, 6]}
    }
}

# Update configs
for exp_name, monkey_settings in MONKEY_CONFIG.items():
    if exp_name in ANALYSIS_CONFIGS:
        ANALYSIS_CONFIGS[exp_name]["short_name"] = monkey_settings["name"]
        ANALYSIS_CONFIGS[exp_name]["show"] = monkey_settings["show"]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def remove_outliers(data, method='iqr_conservative', upper_limit=1000):
    """Remove extreme outliers - MATCHES analysis script method exactly"""
    if len(data) == 0:
        return data
    
    if method == 'iqr_conservative':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3.0 * IQR  # SAME as analysis script
        upper_bound = Q3 + 3.0 * IQR  # SAME as analysis script
        upper_bound = min(upper_bound, upper_limit)  # SAME cap method
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'physiological':
        outlier_mask = (data < 0) | (data > upper_limit)
        
    elif method == 'percentile':
        lower_bound = data.quantile(0.01)
        upper_bound = data.quantile(0.99)
        upper_bound = min(upper_bound, upper_limit)
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    filtered = data[~outlier_mask]
    
    removed = len(data) - len(filtered)
    if removed > 0:
        print(f"      Removed {removed}/{len(data)} outliers (method={method}, limit={upper_limit} µV)")
        print(f"      Range: {data.min():.1f}-{data.max():.1f} → {filtered.min():.1f}-{filtered.max():.1f} µV")
    
    return filtered

def configure_axis_ticks(ax, y_max):
    """Configure axis ticks with proper styling"""
    tick_interval = PLOT_SETTINGS["y_tick_interval"]
    y_max_rounded = int(np.ceil(y_max / tick_interval) * tick_interval)
    
    major_ticks = np.arange(0, y_max_rounded + tick_interval, tick_interval)
    ax.set_yticks(major_ticks)
    
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis='y', direction='out', length=8, width=2,
                   labelsize=PLOT_SETTINGS["tick_label_font_size"], pad=10)
    
    for label in ax.get_yticklabels():
        label.set_fontweight('normal')
        label.set_fontfamily('Arial')
    
    ax.set_ylim(0, y_max_rounded)
    return y_max_rounded

def get_muscle_color(muscle_name):
    """Get the color for a specific muscle"""
    return MUSCLE_COLORS.get(muscle_name, "#FF69B4")

def get_muscle_abbreviation(muscle_name):
    """Get abbreviated name for muscle labels"""
    abbreviations = {
        "Bicep": "Biceps",
        "Brachioradialis": "Brach",
        "Abductor Pollicis Brevis": "APB"
    }
    return abbreviations.get(muscle_name, muscle_name)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_experiment_data(base_path):
    """Load and clean data from all experiments"""
    
    print("📊 LOADING EXPERIMENT DATA")
    print("=" * 50)
    
    all_experiments = {}
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    filtered_configs = {k: v for k, v in ANALYSIS_CONFIGS.items() if v.get('show', True)}
    
    for exp_name, config in filtered_configs.items():
        exp_path = base_path / config['data_path']
        
        if exp_path.exists():
            print(f"📁 Loading {exp_name}...")
            
            df = pd.read_csv(exp_path)
            healthy_data = df[df['hemisphere'] == 'healthy'].copy()
            stroke_data = df[df['hemisphere'] == 'stroke'].copy()
            
            exp_results = {
                'config': config,
                'muscle_data': {},
                'stats_results': {}
            }
            
            for i, (healthy_ch, stroke_ch, muscle_name) in enumerate(zip(
                config['healthy']['channels'],
                config['stroke']['channels'],
                ['Bicep', 'Brachioradialis', 'Abductor Pollicis Brevis']
            )):
                if muscle_name in muscle_names:
                    healthy_col = f'ch{healthy_ch}_amplitude'
                    stroke_col = f'ch{stroke_ch}_amplitude'
                    
                    healthy_raw = healthy_data[healthy_col].dropna() if healthy_col in healthy_data.columns else pd.Series([])
                    stroke_raw = stroke_data[stroke_col].dropna() if stroke_col in stroke_data.columns else pd.Series([])
                    
                    if PLOT_SETTINGS['remove_outliers']:
                        healthy_clean = remove_outliers(healthy_raw, 
                                                       PLOT_SETTINGS['outlier_method'],
                                                       PLOT_SETTINGS['outlier_upper_limit'])
                        stroke_clean = remove_outliers(stroke_raw,
                                                      PLOT_SETTINGS['outlier_method'], 
                                                      PLOT_SETTINGS['outlier_upper_limit'])
                    else:
                        healthy_clean = healthy_raw.copy()
                        stroke_clean = stroke_raw.copy()
                    
                    # Calculate statistics
                    healthy_stats = {
                        'mean': healthy_clean.mean() if len(healthy_clean) > 0 else 0,
                        'std': healthy_clean.std() if len(healthy_clean) > 0 else 0,
                        'sem': healthy_clean.std() / np.sqrt(len(healthy_clean)) if len(healthy_clean) > 0 else 0,
                        'n': len(healthy_clean)
                    }
                    
                    stroke_stats = {
                        'mean': stroke_clean.mean() if len(stroke_clean) > 0 else 0,
                        'std': stroke_clean.std() if len(stroke_clean) > 0 else 0,
                        'sem': stroke_clean.std() / np.sqrt(len(stroke_clean)) if len(stroke_clean) > 0 else 0,
                        'n': len(stroke_clean)
                    }
                    
                    exp_results['muscle_data'][muscle_name] = {
                        'healthy': healthy_clean,
                        'stroke': stroke_clean
                    }
                    
                    exp_results['stats_results'][muscle_name] = {
                        'healthy': healthy_stats,
                        'stroke': stroke_stats
                    }
                    
                    print(f"   {muscle_name}: H={len(healthy_clean)}, S={len(stroke_clean)}")
            
            all_experiments[exp_name] = exp_results
            
        else:
            print(f"⚠️  {exp_name} data not found")
    
    return all_experiments

# =============================================================================
# PLOTTING FUNCTIONS - MULTIPLE LAYOUTS
# =============================================================================

def create_side_by_side_comparison(all_experiments, output_dir):
    """OPTION 1: Side-by-side healthy vs stroke (FIXED SPACING)"""
    
    print("\n📊 CREATING SIDE-BY-SIDE COMPARISON (FIXED SPACING)")
    print("=" * 55)
    
    filtered_experiments = {k: v for k, v in all_experiments.items() 
                          if ANALYSIS_CONFIGS[k].get('show', True)}
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    if not filtered_experiments:
        print("❌ No experiments selected")
        return None
    
    # Create figure with proper spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_SETTINGS["figure_width"], 
                                                   PLOT_SETTINGS["figure_height"]))
    
    # Main title - MOVED HIGHER
    fig.suptitle('MEPs in Healthy vs. Stroke-Affected Hemispheres',  # Changed to singular "Hemisphere"
                 fontsize=PLOT_SETTINGS["title_font_size"] + 4,
                 fontweight='bold', 
                 family='Arial',
                 color='black',
                 y=1.02)  # Moved much higher, outside the plot area
    
    n_experiments = len(filtered_experiments)
    n_muscles = len(muscle_names)
    
    # MORE SPACING between elements
    monkey_base_positions = np.arange(n_experiments) * PLOT_SETTINGS["muscle_group_spacing"]
    muscle_width = 1.3  # Increased spacing
    total_muscle_width = (n_muscles - 1) * muscle_width
    muscle_offsets = np.linspace(-total_muscle_width/2, total_muscle_width/2, n_muscles)
    
    # Find y_max
    all_data = []
    for exp_data in filtered_experiments.values():
        for muscle in muscle_names:
            if muscle in exp_data['muscle_data']:
                all_data.extend(exp_data['muscle_data'][muscle]['healthy'].tolist())
                all_data.extend(exp_data['muscle_data'][muscle]['stroke'].tolist())
    
    y_max = max(all_data) * 1.25 if all_data else 100
    
    # Plot both hemispheres
    for ax, hemisphere, title, color in [
        (ax1, 'healthy', 'Healthy Hemispheres', '#2E7D32'),  # Darker green
        (ax2, 'stroke', 'Stroke-Affected Hemispheres', '#C62828')  # Darker red
    ]:
        ax.set_title(title, 
                    fontsize=PLOT_SETTINGS["title_font_size"],
                    fontweight='bold',
                    color='black',
                    family='Arial',
                    pad=20)  # Reduced from 30 since main title is now higher
        
        for monkey_idx, (exp_name, exp_data) in enumerate(filtered_experiments.items()):
            config_exp = exp_data['config']
            short_name = config_exp['short_name']
            monkey_x = monkey_base_positions[monkey_idx]
            
            for muscle_idx, muscle in enumerate(muscle_names):
                if muscle in exp_data['muscle_data'] and muscle in exp_data['stats_results']:
                    data = exp_data['muscle_data'][muscle][hemisphere]
                    stats = exp_data['stats_results'][muscle][hemisphere]
                    muscle_color = get_muscle_color(muscle)
                    
                    if len(data) > 0:
                        x_center = monkey_x + muscle_offsets[muscle_idx]
                        
                        # Plot individual points
                        jitter_width = 0.35
                        jitter = np.random.uniform(-jitter_width, jitter_width, len(data))
                        x_coords = np.full(len(data), x_center) + jitter
                        
                        ax.scatter(x_coords, data,
                                  color=muscle_color, alpha=0.8, s=180,
                                  marker='o', edgecolors=muscle_color,
                                  linewidth=2, zorder=5)
                        
                        # Plot mean line
                        mean_val = stats['mean']
                        mean_line_width = 0.4
                        ax.plot([x_center - mean_line_width, x_center + mean_line_width],
                               [mean_val, mean_val],
                               color='black', linewidth=5,
                               solid_capstyle='round', zorder=10)
                        
                        # Muscle label - SMALLER FONT and better spacing
                        muscle_label = get_muscle_abbreviation(muscle)
                        ax.text(x_center, -y_max * 0.10, muscle_label,
                               ha='center', va='top',
                               fontsize=PLOT_SETTINGS["tick_label_font_size"] - 6,  # Reduced from -2 to -6
                               fontweight='normal', color='black',
                               fontfamily='Arial')
            
            # Monkey name with MORE SPACE - moved further down
            ax.text(monkey_x, -y_max * 0.22, short_name,  # Increased from 0.18 to 0.22
                   ha='center', va='top',
                   fontsize=PLOT_SETTINGS["axis_label_font_size"],
                   fontweight='bold', color='black',
                   fontfamily='Arial')
        
        # Configure axes with INCREASED MARGINS
        margin_ratio = 0.35  # More margin
        total_width = monkey_base_positions[-1] if len(monkey_base_positions) > 0 else PLOT_SETTINGS["muscle_group_spacing"]
        left_margin = total_width * margin_ratio
        right_margin = total_width * margin_ratio
        
        ax.set_xlim(-left_margin, total_width + right_margin)
        configure_axis_ticks(ax, y_max)
        
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylabel('MEP Amplitude (µV)',
                     fontsize=PLOT_SETTINGS["axis_label_font_size"],
                     fontweight='normal',
                     color='black',
                     fontfamily='Arial',
                     labelpad=25)  # INCREASED labelpad
        
        ax.grid(False)  # REMOVED GRID
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_width"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_facecolor('white')
    
    fig.patch.set_facecolor('white')
    
    # CRITICAL: Adjust subplot spacing - more room at top for title, more at bottom for labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, left=0.08, right=0.97, top=0.88, wspace=0.18)  # Reduced top from 0.92 to 0.88 for higher main title
    
    plot_path = output_dir / 'mep_side_by_side_fixed.png'
    plt.savefig(plot_path, dpi=PLOT_SETTINGS["dpi"],
               bbox_inches='tight', facecolor='white', pad_inches=0.5)
    plt.show()
    
    print(f"✅ Side-by-side plot saved: {plot_path}")
    return fig

def create_individual_nhp_comparisons(all_experiments, output_dir):
    """OPTION 2: Separate figure for each NHP (clearest for detailed comparison)"""
    
    print("\n📊 CREATING INDIVIDUAL NHP COMPARISONS")
    print("=" * 50)
    
    filtered_experiments = {k: v for k, v in all_experiments.items() 
                          if ANALYSIS_CONFIGS[k].get('show', True)}
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    for exp_name, exp_data in filtered_experiments.items():
        config_exp = exp_data['config']
        short_name = config_exp['short_name']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        fig.suptitle(f'{short_name}: Healthy vs. Stroke-Affected Hemisphere',  # Changed to singular
                     fontsize=40, fontweight='bold', family='Arial', 
                     color='black', y=1.00)  # Moved higher
        
        # Calculate y_max for this NHP
        all_data = []
        for muscle in muscle_names:
            if muscle in exp_data['muscle_data']:
                all_data.extend(exp_data['muscle_data'][muscle]['healthy'].tolist())
                all_data.extend(exp_data['muscle_data'][muscle]['stroke'].tolist())
        
        y_max = max(all_data) * 1.25 if all_data else 100
        
        muscle_positions = np.arange(len(muscle_names)) * 3.5  # Increased spacing
        
        for ax, hemisphere, title, color in [
            (ax1, 'healthy', 'Healthy Hemisphere', 'black'),  # Changed to singular
            (ax2, 'stroke', 'Stroke-Affected Hemisphere', 'black')  # Changed to singular
        ]:
            ax.set_title(title, fontsize=36, fontweight='bold', 
                        color=color, family='Arial', pad=15)  # Reduced padding
            
            for muscle_idx, muscle in enumerate(muscle_names):
                if muscle in exp_data['muscle_data']:
                    data = exp_data['muscle_data'][muscle][hemisphere]
                    stats = exp_data['stats_results'][muscle][hemisphere]
                    muscle_color = get_muscle_color(muscle)
                    
                    if len(data) > 0:
                        x_pos = muscle_positions[muscle_idx]
                        
                        jitter = np.random.uniform(-0.3, 0.3, len(data))
                        x_coords = x_pos + jitter
                        
                        ax.scatter(x_coords, data, color=muscle_color, 
                                  alpha=0.8, s=200, edgecolors=muscle_color, 
                                  linewidth=2, zorder=5)
                        
                        mean_val = stats['mean']
                        ax.plot([x_pos - 0.4, x_pos + 0.4], [mean_val, mean_val],
                               color='black', linewidth=5, zorder=10)
                        
                        muscle_label = get_muscle_abbreviation(muscle)
                        ax.text(x_pos, -y_max * 0.10, muscle_label,
                               ha='center', va='top', fontsize=24,  # Reduced from 28
                               fontweight='normal', fontfamily='Arial', color='black')  # Changed to normal weight
            
            ax.set_xlim(-1.5, muscle_positions[-1] + 1.5)
            configure_axis_ticks(ax, y_max)
            
            ax.set_xticks([])
            ax.set_ylabel('MEP Amplitude (µV)', fontsize=34, 
                         fontweight='normal', fontfamily='Arial', labelpad=20,
                         color='black')
            
            ax.grid(False)  # REMOVED GRID
            ax.set_axisbelow(True)
            
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_edgecolor('black')  # Ensure black spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_facecolor('white')
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, left=0.08, right=0.97, top=0.88, wspace=0.15)  # Reduced top for higher title
        
        plot_path = output_dir / f'{short_name}_healthy_vs_stroke.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', pad_inches=0.5)
        plt.show()
        
        print(f"✅ {short_name} comparison saved: {plot_path}")

def create_grouped_by_muscle(all_experiments, output_dir):
    """OPTION 3: Group by muscle type (shows pattern across muscles)"""
    
    print("\n📊 CREATING MUSCLE-GROUPED COMPARISON")
    print("=" * 50)
    
    filtered_experiments = {k: v for k, v in all_experiments.items() 
                          if ANALYSIS_CONFIGS[k].get('show', True)}
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    
    fig.suptitle('MEP Responses by Muscle Group', 
                 fontsize=42, fontweight='bold', family='Arial', 
                 color='black', y=1.00)  # Moved higher
    
    for muscle_idx, (muscle, ax) in enumerate(zip(muscle_names, axes)):
        muscle_label = get_muscle_abbreviation(muscle)
        muscle_color = get_muscle_color(muscle)
        
        ax.set_title(muscle_label, fontsize=38, fontweight='bold',
                    color='black', family='Arial', pad=15)  # Reduced padding
        
        # Collect all data for this muscle
        all_muscle_data = []
        for exp_data in filtered_experiments.values():
            if muscle in exp_data['muscle_data']:
                all_muscle_data.extend(exp_data['muscle_data'][muscle]['healthy'].tolist())
                all_muscle_data.extend(exp_data['muscle_data'][muscle]['stroke'].tolist())
        
        y_max = max(all_muscle_data) * 1.25 if all_muscle_data else 100
        
        positions = np.arange(len(filtered_experiments)) * 3
        
        for nhp_idx, (exp_name, exp_data) in enumerate(filtered_experiments.items()):
            config_exp = exp_data['config']
            short_name = config_exp['short_name']
            
            if muscle in exp_data['muscle_data']:
                healthy_data = exp_data['muscle_data'][muscle]['healthy']
                stroke_data = exp_data['muscle_data'][muscle]['stroke']
                
                healthy_stats = exp_data['stats_results'][muscle]['healthy']
                stroke_stats = exp_data['stats_results'][muscle]['stroke']
                
                x_healthy = positions[nhp_idx] - 0.6
                x_stroke = positions[nhp_idx] + 0.6
                
                # Plot healthy
                if len(healthy_data) > 0:
                    jitter = np.random.uniform(-0.2, 0.2, len(healthy_data))
                    ax.scatter(x_healthy + jitter, healthy_data,
                              color='#4CAF50', alpha=0.8, s=180, 
                              edgecolors='#2E7D32', linewidth=2, zorder=5)
                    ax.plot([x_healthy - 0.3, x_healthy + 0.3], 
                           [healthy_stats['mean'], healthy_stats['mean']],
                           color='black', linewidth=5, zorder=10)
                
                # Plot stroke
                if len(stroke_data) > 0:
                    jitter = np.random.uniform(-0.2, 0.2, len(stroke_data))
                    ax.scatter(x_stroke + jitter, stroke_data,
                              color='#EF5350', alpha=0.8, s=180,
                              edgecolors='#C62828', linewidth=2, zorder=5)
                    ax.plot([x_stroke - 0.3, x_stroke + 0.3],
                           [stroke_stats['mean'], stroke_stats['mean']],
                           color='black', linewidth=5, zorder=10)
                
                # Labels - SMALLER and better spaced
                ax.text(x_healthy, -y_max * 0.10, 'H', ha='center', va='top',
                       fontsize=22, fontweight='bold', color='black',  # Reduced from 24
                       fontfamily='Arial')
                ax.text(x_stroke, -y_max * 0.10, 'S', ha='center', va='top',
                       fontsize=22, fontweight='bold', color='black',  # Reduced from 24
                       fontfamily='Arial')
                
                ax.text(positions[nhp_idx], -y_max * 0.20, short_name,
                       ha='center', va='top', fontsize=28, fontweight='bold',  # Reduced from 32
                       fontfamily='Arial', color='black')
        
        ax.set_xlim(-1.5, positions[-1] + 1.5)
        configure_axis_ticks(ax, y_max)
        
        ax.set_xticks([])
        if muscle_idx == 0:
            ax.set_ylabel('MEP Amplitude (µV)', fontsize=34,
                         fontweight='normal', fontfamily='Arial', labelpad=20,
                         color='black')  # Changed to black
        
        ax.grid(False)  # REMOVED GRID
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_edgecolor('black')  # Ensure black spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_facecolor('white')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24, left=0.06, right=0.97, top=0.86, wspace=0.12)  # Reduced top for higher title
    
    plot_path = output_dir / 'mep_grouped_by_muscle.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight',
               facecolor='white', pad_inches=0.5)
    plt.show()
    
    print(f"✅ Muscle-grouped plot saved: {plot_path}")

# =============================================================================
# STATISTICS PRINTING
# =============================================================================

def print_comparison_statistics(all_experiments):
    """Print detailed statistics"""
    
    print("\n" + "="*70)
    print("DETAILED MEP COMPARISON STATISTICS")
    print("="*70)
    
    filtered_experiments = {k: v for k, v in all_experiments.items() 
                          if ANALYSIS_CONFIGS[k].get('show', True)}
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    for exp_name, exp_data in filtered_experiments.items():
        config_exp = exp_data['config']
        print(f"\n🐒 {config_exp['short_name']}:")
        print("-" * 60)
        
        for muscle in muscle_names:
            if muscle in exp_data['stats_results']:
                h_stats = exp_data['stats_results'][muscle]['healthy']
                s_stats = exp_data['stats_results'][muscle]['stroke']
                
                impairment = (1 - s_stats['mean'] / h_stats['mean']) * 100 if h_stats['mean'] > 0 else 0
                
                print(f"\n  {muscle}:")
                print(f"    Healthy:  {h_stats['mean']:.1f} ± {h_stats['sem']:.1f} µV (n={h_stats['n']})")
                print(f"    Stroke:   {s_stats['mean']:.1f} ± {s_stats['sem']:.1f} µV (n={s_stats['n']})")
                print(f"    Impairment: {impairment:.1f}%")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MEP Comparison - MATCHES Analysis Script Methods')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Base path containing experiment folders')
    parser.add_argument('--output-dir', type=Path, help='Output directory')
    parser.add_argument('--layout', type=str, default='all',
                       choices=['side-by-side', 'individual', 'muscle-grouped', 'all'],
                       help='Which layout to create')
    parser.add_argument('--outlier-method', type=str, default='iqr_conservative',
                       choices=['iqr_conservative', 'physiological', 'percentile'],
                       help='Outlier removal method (default: iqr_conservative) - MATCHES analysis script')
    parser.add_argument('--outlier-limit', type=float, default=1000,
                       help='Upper limit for outlier removal (µV) - default 1000 (MATCHES analysis script)')
    parser.add_argument('--separate-scales', action='store_true',
                       help='Use separate y-axis scales for healthy vs stroke')
    
    args = parser.parse_args()
    
    # Update settings based on arguments
    PLOT_SETTINGS['outlier_method'] = args.outlier_method
    PLOT_SETTINGS['outlier_upper_limit'] = args.outlier_limit
    PLOT_SETTINGS['use_separate_y_scales'] = args.separate_scales
    
    if not args.output_dir:
        args.output_dir = args.base_path / 'mep_comparisons'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("  MEP COMPARISON - USING SAME METHODS AS ANALYSIS SCRIPT")
    print("="*70)
    print(f"📁 Output: {args.output_dir}")
    print(f"📐 Layout: {args.layout}")
    print(f"🧹 Outlier method: {args.outlier_method}")
    print(f"🧹 Outlier limit: {args.outlier_limit} µV")
    print(f"📊 Separate y-scales: {args.separate_scales}")
    print(f"✅ Methods match analysis script for reproducibility")
    
    # Configure matplotlib
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': PLOT_SETTINGS["tick_label_font_size"],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    
    try:
        all_experiments = load_all_experiment_data(args.base_path)
        
        if not all_experiments:
            print("❌ No experiment data found")
            return
        
        print(f"\n✅ Successfully loaded {len(all_experiments)} experiments")
        
        # Create requested layouts
        if args.layout == 'side-by-side' or args.layout == 'all':
            create_side_by_side_comparison(all_experiments, args.output_dir)
        
        if args.layout == 'individual' or args.layout == 'all':
            create_individual_nhp_comparisons(all_experiments, args.output_dir)
        
        if args.layout == 'muscle-grouped' or args.layout == 'all':
            create_grouped_by_muscle(all_experiments, args.output_dir)
        
        print_comparison_statistics(all_experiments)
        
        print(f"\n✅ COMPARISON PLOTS COMPLETE!")
        print(f"\n📊 METHODOLOGY:")
        print(f"   • Outlier removal: {args.outlier_method} method")
        print(f"   • Upper limit: {args.outlier_limit} µV")
        print(f"   • Matches analysis script settings for consistency")
        print(f"\n📊 LAYOUT RECOMMENDATIONS:")
        print(f"   • Side-by-side: Best for overall comparison, good for papers")
        print(f"   • Individual NHPs: Best for detailed per-subject analysis")
        print(f"   • Muscle-grouped: Best for showing patterns across muscles")
        print(f"\n💡 TROUBLESHOOTING:")
        print(f"   If plot is still compressed by outliers:")
        print(f"   1. Try: python script.py --outlier-limit 200")
        print(f"   2. Try: python script.py --separate-scales")
        print(f"   3. Try: python script.py --outlier-method physiological --outlier-limit 200")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()