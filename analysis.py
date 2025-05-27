#!/usr/bin/env python3
"""
Enhanced Unified MEP Analysis Script - All Experiments
Analyzes individual experiments and creates customizable grouped comparison charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import argparse
import json

# =============================================================================
# 🎯 EASY CONFIGURATION SECTION - EDIT HERE TO CUSTOMIZE YOUR PLOTS
# =============================================================================

# MONKEY CONFIGURATION - Set to True to show, False to hide
MONKEY_CONFIG = {
    "Nov5_Olive": {
        "show": True,           # Set to False to hide this monkey
        "color": "#FF6B6B",     # Change color here (red)
        "name": "Olive",        # Display name
        "mean_line_color": "black",     # Color of mean line
        "mean_line_width": 5,           # Thickness of mean line
        "mean_line_length": 0.25        # Length of mean line (0.1 = short, 0.5 = long)
    },
    "Nov5_Chive": {
        "show": False,          # EXAMPLE: Hide Chive
        "color": "#4ECDC4",     # Change color here (teal)
        "name": "Chive",        # Display name
        "mean_line_color": "black",     
        "mean_line_width": 5,           
        "mean_line_length": 0.25        
    },
    "Nov5_Cheddar": {
        "show": True,           # Set to False to hide this monkey
        "color": "#45B7D1",     # Change color here (blue)
        "name": "Cheddar",      # Display name
        "mean_line_color": "black",     
        "mean_line_width": 5,           
        "mean_line_length": 0.25        
    },
    "Oct31_Chive": {
        "show": True,           # Set to False to hide this monkey
        "color": "#9B59B6",     # Change color here (purple)
        "name": "Chive", # Display name
        "mean_line_color": "black",     
        "mean_line_width": 5,           
        "mean_line_length": 0.25        
    }
}

# PLOT SETTINGS
PLOT_SETTINGS = {
    "show_legend": False,           # Set to False to hide legend in grouped plots
    "figure_width": 16,             # Width of individual plots
    "figure_height": 8,             # Height of individual plots
    "point_size": 25,               # Size of scatter points
    "mean_line_thickness": 5,       # Thickness of mean lines
    "text_padding": 0.15,           # Padding around monkey labels (increase to reduce overlap)
    "remove_outliers": True,        # Set to False to keep all data points (no outlier removal)
    "outlier_method": "iqr_conservative",  # Method: 'iqr_conservative', 'physiological', 'percentile'
    "outlier_upper_limit": 1000,    # Upper limit for MEP values (µV)
    "save_stats_csv": True,         # Set to True to save statistics to CSV file
    "save_stats_txt": True,         # Set to True to save detailed report to text file
    "y_axis_minor_ticks": 10,       # NEW: Y-axis minor tick interval (µV)
    "y_axis_major_ticks": 20,       # NEW: Y-axis major tick interval (µV) 
    "muscle_group_spacing": 2.5,    # NEW: Spacing between muscle groups (reduced from 3.5)
}

# MUSCLE CONFIGURATION - Set to True to show, False to hide
MUSCLE_CONFIG = {
    "Upper Arm": True,
    "Forearm": True,
    "Hand": True
}

# =============================================================================
# END OF EASY CONFIGURATION SECTION
# =============================================================================

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

# Experiment configurations with data paths
ANALYSIS_CONFIGS = {
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "short_name": "Olive",  # Will be updated from MONKEY_CONFIG
        "color": "#FF6B6B",     # Will be updated from MONKEY_CONFIG
        "show": True,           # Will be updated from MONKEY_CONFIG
        "data_path": "Nov5_Olive/mep_results.csv",
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
        "short_name": "Chive",
        "color": "#4ECDC4",
        "show": True,
        "data_path": "Nov5_Chive/mep_results.csv",
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
        "short_name": "Cheddar",
        "color": "#45B7D1",
        "show": True,
        "data_path": "Nov5_Cheddar/mep_results.csv",
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
        "short_name": "Oct31 Chive",
        "color": "#9B59B6",
        "show": True,
        "data_path": "Oct31_Chive/mep_results.csv",
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

# UPDATE ANALYSIS_CONFIGS with MONKEY_CONFIG settings
for exp_name, monkey_settings in MONKEY_CONFIG.items():
    if exp_name in ANALYSIS_CONFIGS:
        ANALYSIS_CONFIGS[exp_name]["short_name"] = monkey_settings["name"]
        ANALYSIS_CONFIGS[exp_name]["color"] = monkey_settings["color"]
        ANALYSIS_CONFIGS[exp_name]["show"] = monkey_settings["show"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_color"] = monkey_settings["mean_line_color"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_width"] = monkey_settings["mean_line_width"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_length"] = monkey_settings["mean_line_length"]

# Default color palettes for customization
COLOR_PALETTES = {
    'default': ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6", "#F39C12", "#2ECC71"],
    'viridis': ["#440154", "#31688e", "#35b779", "#fde725"],
    'plasma': ["#0d0887", "#7e03a8", "#cc4778", "#f0f921"],
    'colorblind': ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    'pastel': ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA", "#E0BBE4"],
    'bold': ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
}

def set_detailed_y_axis(ax, y_max, minor_tick_interval=10, major_tick_interval=20):
    """Set detailed y-axis with intelligent adaptive ticking"""
    
    # Calculate nice upper limit rounded to major tick interval
    nice_y_max = np.ceil(y_max / major_tick_interval) * major_tick_interval
    
    # ADAPTIVE TICK SYSTEM - Adjust intervals based on data range
    if nice_y_max > 200:
        # For large ranges, use larger intervals to avoid overcrowding
        if nice_y_max > 500:
            major_interval = 100  # Every 100µV for very large ranges
            minor_interval = 50   # Every 50µV
        else:
            major_interval = 50   # Every 50µV for medium-large ranges
            minor_interval = 25   # Every 25µV
        # Recalculate nice_y_max with new interval
        nice_y_max = np.ceil(y_max / major_interval) * major_interval
    elif nice_y_max > 100:
        # For medium ranges, use moderate intervals
        major_interval = major_tick_interval  # Use config setting (default 20µV)
        minor_interval = minor_tick_interval  # Use config setting (default 10µV)
    else:
        # For small ranges, use fine intervals
        major_interval = 20   # Every 20µV
        minor_interval = 10   # Every 10µV
    
    # Generate tick arrays
    major_ticks = np.arange(0, nice_y_max + major_interval, major_interval)
    minor_ticks = np.arange(0, nice_y_max + minor_interval, minor_interval)
    
    # Limit the number of ticks to prevent overcrowding
    max_major_ticks = 12  # Maximum number of major ticks
    max_minor_ticks = 25  # Maximum number of minor ticks
    
    if len(major_ticks) > max_major_ticks:
        # Reduce tick density by doubling the interval
        major_interval *= 2
        major_ticks = np.arange(0, nice_y_max + major_interval, major_interval)
    
    if len(minor_ticks) > max_minor_ticks:
        # Reduce minor tick density
        minor_interval *= 2
        minor_ticks = np.arange(0, nice_y_max + minor_interval, minor_interval)
    
    # Set the ticks
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    # Style the ticks
    ax.tick_params(axis='y', which='major', labelsize=12, length=6, width=1)
    ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
    # Update y-limit to the nice maximum
    ax.set_ylim(0, nice_y_max)
    
    # Add grid for both major and minor ticks
    ax.grid(True, which='major', alpha=0.4, linestyle='-', color='gray', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.2, linestyle='-', color='gray', linewidth=0.3)
    
    return nice_y_max

def remove_extreme_outliers(data: pd.Series, method='iqr_conservative', upper_limit=1000) -> tuple:
    """Remove extreme outliers using various methods"""
    
    if len(data) == 0:
        return data, pd.Series([], dtype=bool), {}
    
    original_count = len(data)
    
    if method == 'iqr_conservative':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        upper_bound = min(upper_bound, upper_limit)
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'physiological':
        outlier_mask = (data < 0) | (data > upper_limit)
        
    elif method == 'percentile':
        lower_bound = data.quantile(0.01)
        upper_bound = data.quantile(0.99)
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

def load_all_experiment_data(base_path: Path, outlier_method=None, upper_limit=None):
    """Load and clean data from all experiments - FILTERED BY CONFIG"""
    
    # Use config settings if parameters not provided
    if outlier_method is None:
        outlier_method = PLOT_SETTINGS["outlier_method"]
    if upper_limit is None:
        upper_limit = PLOT_SETTINGS["outlier_upper_limit"]
    
    print("📊 LOADING EXPERIMENT DATA (FILTERED BY CONFIG)")
    print("=" * 50)
    
    all_experiments = {}
    # Filter muscles based on config
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    # Filter experiments based on config
    filtered_configs = {k: v for k, v in ANALYSIS_CONFIGS.items() if v.get('show', True)}
    
    print(f"🐒 Active monkeys: {list(filtered_configs.keys())}")
    print(f"💪 Active muscles: {muscle_names}")
    print(f"🧹 Remove outliers: {PLOT_SETTINGS['remove_outliers']}")
    if PLOT_SETTINGS['remove_outliers']:
        print(f"   Method: {outlier_method}, Upper limit: {upper_limit}µV")
    
    for exp_name, config in filtered_configs.items():
        exp_path = base_path / config['data_path']
        
        if exp_path.exists():
            print(f"📁 Loading {exp_name}...")
            
            df = pd.read_csv(exp_path)
            healthy_data = df[df['hemisphere'] == 'healthy'].copy()
            stroke_data = df[df['hemisphere'] == 'stroke'].copy()
            
            print(f"   ✅ {len(healthy_data)} healthy + {len(stroke_data)} stroke MEPs")
            
            exp_results = {
                'config': config,
                'muscle_data': {},
                'cleaning_report': {},
                'stats_results': {}
            }
            
            # Process each active muscle
            for i, (healthy_ch, stroke_ch, muscle_name) in enumerate(zip(
                config['healthy']['channels'],
                config['stroke']['channels'],
                ['Upper Arm', 'Forearm', 'Hand']  # All muscle names
            )):
                # Only process if muscle is active in config
                if muscle_name in muscle_names:
                    healthy_col = f'ch{healthy_ch}_amplitude'
                    stroke_col = f'ch{stroke_ch}_amplitude'
                    
                    # Get raw data
                    healthy_raw = healthy_data[healthy_col].dropna() if healthy_col in healthy_data.columns else pd.Series([])
                    stroke_raw = stroke_data[stroke_col].dropna() if stroke_col in stroke_data.columns else pd.Series([])
                    
                    # Clean outliers ONLY if config says to
                    if PLOT_SETTINGS['remove_outliers']:
                        healthy_clean, _, healthy_stats = remove_extreme_outliers(healthy_raw, outlier_method, upper_limit)
                        stroke_clean, _, stroke_stats = remove_extreme_outliers(stroke_raw, outlier_method, upper_limit)
                    else:
                        # Keep all data - no outlier removal
                        healthy_clean = healthy_raw.copy()
                        stroke_clean = stroke_raw.copy()
                        healthy_stats = {
                            'original_count': len(healthy_raw),
                            'removed_count': 0,
                            'percent_removed': 0,
                            'method': 'none',
                            'upper_limit': None,
                            'final_range': (healthy_raw.min(), healthy_raw.max()) if len(healthy_raw) > 0 else (0, 0)
                        }
                        stroke_stats = {
                            'original_count': len(stroke_raw),
                            'removed_count': 0,
                            'percent_removed': 0,
                            'method': 'none',
                            'upper_limit': None,
                            'final_range': (stroke_raw.min(), stroke_raw.max()) if len(stroke_raw) > 0 else (0, 0)
                        }
                    
                    exp_results['muscle_data'][muscle_name] = {
                        'healthy': healthy_clean,
                        'stroke': stroke_clean,
                        'healthy_raw': healthy_raw,
                        'stroke_raw': stroke_raw,
                        'healthy_channel': healthy_ch,
                        'stroke_channel': stroke_ch,
                        'healthy_name': config['healthy']['channel_names'][i],
                        'stroke_name': config['stroke']['channel_names'][i]
                    }
                    
                    exp_results['cleaning_report'][muscle_name] = {
                        'healthy': healthy_stats,
                        'stroke': stroke_stats
                    }
                    
                    print(f"   {muscle_name}: H={len(healthy_clean)}/{len(healthy_raw)}, S={len(stroke_clean)}/{len(stroke_raw)}")
            
            # Calculate statistics for this experiment
            exp_results['stats_results'] = calculate_experiment_statistics(exp_results['muscle_data'])
            all_experiments[exp_name] = exp_results
            
        else:
            print(f"⚠️ {exp_name} data not found at {exp_path}")
    
    return all_experiments

def calculate_experiment_statistics(muscle_data: dict) -> dict:
    """Calculate statistics for a single experiment"""
    
    stats_results = {}
    
    for muscle, data in muscle_data.items():
        healthy = data['healthy']
        stroke = data['stroke']
        
        if len(healthy) == 0 or len(stroke) == 0:
            continue
        
        # Basic statistics
        healthy_stats = {
            'mean': healthy.mean(),
            'std': healthy.std(),
            'sem': healthy.std() / np.sqrt(len(healthy)),
            'median': healthy.median(),
            'n': len(healthy)
        }
        
        stroke_stats = {
            'mean': stroke.mean(),
            'std': stroke.std(),
            'sem': stroke.std() / np.sqrt(len(stroke)),
            'median': stroke.median(),
            'n': len(stroke)
        }
        
        # Statistical test
        try:
            statistic, p_value = stats.mannwhitneyu(healthy, stroke, alternative='two-sided')
        except:
            statistic, p_value = 0, 1.0
        
        # Impairment calculations
        mean_impairment = (1 - stroke_stats['mean'] / healthy_stats['mean']) * 100 if healthy_stats['mean'] > 0 else 0
        
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
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'healthy_name': data['healthy_name'],
            'stroke_name': data['stroke_name']
        }
    
    return stats_results

def create_individual_experiment_plots(all_experiments: dict, output_dir: Path):
    """Create individual plots for each experiment with enhanced y-axis detail"""
    
    print("\n📊 CREATING INDIVIDUAL EXPERIMENT PLOTS")
    print("=" * 45)
    
    muscle_names = ['Upper Arm', 'Forearm', 'Hand']
    muscle_positions = [1, 2, 3]
    
    for exp_name, exp_data in all_experiments.items():
        config = exp_data['config']
        muscle_data = exp_data['muscle_data']
        stats_results = exp_data['stats_results']
        cleaning_report = exp_data['cleaning_report']
        
        print(f"   Creating plots for {exp_name}...")
        
        # Create figure for this experiment using config settings
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
        fig.suptitle(f'MEP Analysis: {config["description"]}', 
                     fontsize=18, fontweight='bold')
        
        # Calculate y-limits
        all_healthy_data = []
        all_stroke_data = []
        
        for muscle in muscle_names:
            if muscle in muscle_data:
                all_healthy_data.extend(muscle_data[muscle]['healthy'].tolist())
                all_stroke_data.extend(muscle_data[muscle]['stroke'].tolist())
        
        healthy_y_max = max(all_healthy_data) * 1.15 if all_healthy_data else 100
        stroke_y_max = max(all_stroke_data) * 1.15 if all_stroke_data else 100
        
        # Plot 1: Healthy Hemisphere
        ax1.set_title('Healthy Hemisphere', fontsize=16, fontweight='bold', color='green', pad=20)
        
        for i, muscle in enumerate(muscle_names):
            if muscle in muscle_data and muscle in stats_results:
                healthy_data = muscle_data[muscle]['healthy']
                stats = stats_results[muscle]['healthy']
                
                if len(healthy_data) > 0:
                    x_pos = muscle_positions[i]
                    jitter = np.random.normal(0, 0.06, len(healthy_data))
                    x_coords = np.full(len(healthy_data), x_pos) + jitter
                    
                    # Plot individual points using config settings
                    ax1.scatter(x_coords, healthy_data, alpha=0.6, s=PLOT_SETTINGS["point_size"], 
                               color='green', edgecolors='white', linewidth=0.5)
                    
                    # Plot mean line using config settings
                    mean_line_width = 0.15
                    ax1.plot([x_pos - mean_line_width, x_pos + mean_line_width], 
                            [stats['mean'], stats['mean']], 
                            color='black', linewidth=PLOT_SETTINGS["mean_line_thickness"], 
                            solid_capstyle='round', zorder=10)
                    
                    # REMOVED sample size info and outlier removal text
                    # No text below x-axis for individual plots
        
        ax1.set_xlim(0.5, 3.5)
        # Apply detailed y-axis formatting
        set_detailed_y_axis(ax1, healthy_y_max, 
                           PLOT_SETTINGS["y_axis_minor_ticks"], 
                           PLOT_SETTINGS["y_axis_major_ticks"])
        
        ax1.set_xticks(muscle_positions)
        ax1.set_xticklabels(muscle_names, fontsize=12, fontweight='bold')
        ax1.set_xlabel('Muscle Groups', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MEP Amplitude (µV)', fontsize=12, fontweight='bold')
        
        # Plot 2: Stroke Hemisphere
        ax2.set_title('Stroke Hemisphere', fontsize=16, fontweight='bold', color='red', pad=20)
        
        for i, muscle in enumerate(muscle_names):
            if muscle in muscle_data and muscle in stats_results:
                stroke_data = muscle_data[muscle]['stroke']
                stats = stats_results[muscle]['stroke']
                
                if len(stroke_data) > 0:
                    x_pos = muscle_positions[i]
                    jitter = np.random.normal(0, 0.06, len(stroke_data))
                    x_coords = np.full(len(stroke_data), x_pos) + jitter
                    
                    # Plot individual points using config settings
                    ax2.scatter(x_coords, stroke_data, alpha=0.6, s=PLOT_SETTINGS["point_size"], 
                               color='red', edgecolors='white', linewidth=0.5)
                    
                    # Plot mean line using config settings
                    mean_line_width = 0.15
                    ax2.plot([x_pos - mean_line_width, x_pos + mean_line_width], 
                            [stats['mean'], stats['mean']], 
                            color='black', linewidth=PLOT_SETTINGS["mean_line_thickness"], 
                            solid_capstyle='round', zorder=10)
                    
                    # REMOVED sample size info and outlier removal text
                    # No text below x-axis for individual plots
        
        ax2.set_xlim(0.5, 3.5)
        # Apply detailed y-axis formatting
        set_detailed_y_axis(ax2, stroke_y_max, 
                           PLOT_SETTINGS["y_axis_minor_ticks"], 
                           PLOT_SETTINGS["y_axis_major_ticks"])
        
        ax2.set_xticks(muscle_positions)
        ax2.set_xticklabels(muscle_names, fontsize=12, fontweight='bold')
        ax2.set_xlabel('Muscle Groups', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MEP Amplitude (µV)', fontsize=12, fontweight='bold')
        
        # Add significance stars
        for i, muscle in enumerate(muscle_names):
            if muscle in stats_results and stats_results[muscle]['significant']:
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
        
        # Save individual plot
        plot_path = output_dir / f'{config["short_name"]}_individual_mep_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"   ✅ {exp_name} plot saved: {plot_path}")

def create_customizable_comparison_plot(all_experiments: dict, output_dir: Path, 
                                      selected_experiments=None, selected_muscles=None,
                                      color_palette='default', custom_colors=None,
                                      custom_labels=None, plot_config=None):
    """Create customizable comparison plot with selection options"""
    
    print("\n📊 CREATING CUSTOMIZABLE COMPARISON PLOT")
    print("=" * 48)
    
    # Default configuration - NOW DEFAULTS TO SEPARATE FIGURES
    default_config = {
        'figure_size': (PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]),
        'title_size': 20,
        'axis_title_size': 18,
        'axis_label_size': 16,
        'tick_label_size': 12,
        'legend_size': 12,
        'point_size': PLOT_SETTINGS["point_size"],
        'point_alpha': 0.7,
        'mean_line_width': PLOT_SETTINGS["mean_line_thickness"],
        'grid_alpha': 0.4,
        'show_mean_values': True,
        'show_sample_sizes': True,
        'show_experiment_names': True,
        'show_statistics_box': True,
        'separate_figures': True,
        'mean_annotation_size': 10,
        'sample_size_text_size': 9,
        'experiment_name_size': 10,
        'jitter_width': 0.25,
        'mean_line_width_individual': 0.3
    }
    
    # Update with user config
    if plot_config:
        default_config.update(plot_config)
    config = default_config
    
    # Filter experiments and muscles based on CONFIG - FIXED FILTERING
    if selected_experiments is None:
        # Only include experiments where show=True in config
        selected_experiments = [k for k, v in ANALYSIS_CONFIGS.items() if v.get('show', True)]
    if selected_muscles is None:
        selected_muscles = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    # Filter data - ENSURE we respect the show setting
    filtered_experiments = {}
    for k, v in all_experiments.items():
        if k in selected_experiments and ANALYSIS_CONFIGS[k].get('show', True):
            filtered_experiments[k] = v
    
    if not filtered_experiments:
        print("❌ No experiments selected or found")
        return None
    
    print(f"   Selected experiments: {list(filtered_experiments.keys())}")
    print(f"   Selected muscles: {selected_muscles}")
    
    # Set up labels
    if custom_labels:
        for exp_name, label in custom_labels.items():
            if exp_name in filtered_experiments:
                filtered_experiments[exp_name]['config']['short_name'] = label
    
    # Collect all data for y-limit calculation
    all_data = []
    for exp_data in filtered_experiments.values():
        muscle_data = exp_data['muscle_data']
        for muscle in selected_muscles:
            if muscle in muscle_data:
                all_data.extend(muscle_data[muscle]['healthy'].tolist())
                all_data.extend(muscle_data[muscle]['stroke'].tolist())
    
    y_max = max(all_data) * 1.15 if all_data else 100
    
    # Create figure(s)
    if config['separate_figures']:
        # Create separate figures for healthy and stroke
        return create_separate_figures(filtered_experiments, selected_muscles, config, y_max, output_dir)
    else:
        # Create combined figure
        return create_combined_figure(filtered_experiments, selected_muscles, config, y_max, output_dir)

def plot_clean_hemisphere_data(ax, filtered_experiments, selected_muscles, hemisphere, 
                              muscle_base_positions, exp_offsets, config, y_max):
    """Plot data with clean, organized layout with wider, dynamic scatter plots"""
    
    for muscle_idx, muscle in enumerate(selected_muscles):
        muscle_x = muscle_base_positions[muscle_idx]
        
        for exp_idx, (exp_name, exp_data) in enumerate(filtered_experiments.items()):
            config_exp = exp_data['config']
            exp_color = config_exp['color']
            short_name = config_exp['short_name']
            
            if muscle in exp_data['muscle_data']:
                data_values = exp_data['muscle_data'][muscle][hemisphere]
                
                if len(data_values) > 0:
                    # Position for this experiment
                    x_center = muscle_x + exp_offsets[exp_idx]
                    
                    # WIDER jitter for better fill - dynamic based on available space
                    jitter_width = 0.20  # Increased from 0.08 to 0.20 for wider spread
                    jitter = np.random.uniform(-jitter_width, jitter_width, len(data_values))
                    x_coords = np.full(len(data_values), x_center) + jitter
                    
                    # Plot scatter points with settings from config
                    ax.scatter(x_coords, data_values, 
                              color=exp_color, alpha=0.6, s=PLOT_SETTINGS["point_size"],
                              edgecolors='white', linewidth=0.3, zorder=5)
                    
                    # Plot CUSTOMIZABLE mean line with individual settings
                    mean_val = data_values.mean()
                    mean_line_width = config_exp.get('mean_line_length', 0.25)  # Use config setting
                    mean_line_color = config_exp.get('mean_line_color', 'black')  # Use config color
                    mean_line_thickness = config_exp.get('mean_line_width', PLOT_SETTINGS["mean_line_thickness"])  # Use config thickness
                    
                    ax.plot([x_center - mean_line_width, x_center + mean_line_width], 
                           [mean_val, mean_val], 
                           color=mean_line_color, linewidth=mean_line_thickness, 
                           solid_capstyle='round', zorder=10)
                    
                    # Add experiment name below x-axis - POSITIONED TO AVOID OVERLAP
                    name_y_pos = -y_max * 0.06  # Fixed position relative to data range
                    ax.text(x_center, name_y_pos, short_name, 
                           ha='center', va='top', fontsize=11, fontweight='bold',
                           color=exp_color)
        
        # Add muscle name BELOW all monkey names for this muscle group
        muscle_name_y_pos = -y_max * 0.12  # Below monkey names
        ax.text(muscle_x, muscle_name_y_pos, muscle, 
               ha='center', va='top', fontsize=14, fontweight='bold',
               color='black')

def create_separate_figures(filtered_experiments, selected_muscles, config, y_max, output_dir):
    """Create separate figures for healthy and stroke hemispheres with enhanced y-axis and closer groups"""
    
    figures = []
    
    for hemisphere, title_color in [('healthy', 'green'), ('stroke', 'red')]:
        # Create figure with settings from config
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
        
        # Enhanced title
        hemisphere_title = f'{hemisphere.capitalize()} Hemisphere'
        ax.set_title(hemisphere_title, fontsize=24, fontweight='bold', color=title_color, pad=30)
        
        n_experiments = len(filtered_experiments)
        n_muscles = len(selected_muscles)
        
        # Create CLOSER positioning with new config setting
        muscle_base_positions = np.arange(n_muscles) * PLOT_SETTINGS["muscle_group_spacing"]  # Now uses config setting
        
        # Position experiments within each muscle group
        exp_width = 0.7  # Wider experiment spacing
        total_exp_width = (n_experiments - 1) * exp_width
        exp_offsets = np.linspace(-total_exp_width/2, total_exp_width/2, n_experiments)
        
        # DYNAMIC Y-MAX calculation for this hemisphere only
        hemisphere_data = []
        for exp_data in filtered_experiments.values():
            muscle_data = exp_data['muscle_data']
            for muscle in selected_muscles:
                if muscle in muscle_data:
                    hemisphere_data.extend(muscle_data[muscle][hemisphere].tolist())
        
        # Calculate y_max for this specific hemisphere and data
        hemisphere_y_max = max(hemisphere_data) * 1.15 if hemisphere_data else 100
        
        # Plot data for each muscle and experiment
        plot_clean_hemisphere_data(ax, filtered_experiments, selected_muscles, hemisphere, 
                                  muscle_base_positions, exp_offsets, config, hemisphere_y_max)
        
        # Configure axis with better space utilization and FIXED PADDING
        margin_ratio = 0.25  # Increased padding
        total_width = muscle_base_positions[-1] if len(muscle_base_positions) > 0 else PLOT_SETTINGS["muscle_group_spacing"]
        left_margin = total_width * margin_ratio
        right_margin = total_width * margin_ratio
        
        # Set limits with PROPER spacing - NO negative y values, but space for text
        ax.set_xlim(-left_margin, total_width + right_margin)
        
        # Apply ENHANCED Y-AXIS formatting with detailed ticks
        nice_y_max = set_detailed_y_axis(ax, hemisphere_y_max, 
                                        PLOT_SETTINGS["y_axis_minor_ticks"], 
                                        PLOT_SETTINGS["y_axis_major_ticks"])
        
        # Set muscle group labels - REMOVED muscle group labels from x-axis
        ax.set_xticks(muscle_base_positions)
        ax.set_xticklabels([''] * len(selected_muscles))  # Empty labels
        # REMOVED x-axis label completely
        ax.set_ylabel('MEP Amplitude (µV)', fontsize=18, fontweight='bold', labelpad=15)
        
        # Add subtle vertical separators between muscle groups
        for i in range(1, len(muscle_base_positions)):
            separator_x = (muscle_base_positions[i-1] + muscle_base_positions[i]) / 2
            ax.axvline(x=separator_x, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
        
        # CONDITIONALLY create legend based on config
        legend = None
        if PLOT_SETTINGS["show_legend"]:
            legend_elements = []
            for exp_name, exp_data in filtered_experiments.items():
                config_exp = exp_data['config']
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=config_exp['color'], markersize=12, alpha=0.9,
                              label=config_exp['short_name'], markeredgecolor='white', markeredgewidth=1)
                )
            
            # Position legend at bottom
            legend = ax.legend(handles=legend_elements, 
                              loc='upper center', 
                              bbox_to_anchor=(0.5, -0.15),  # Further below to avoid text
                              ncol=len(legend_elements),
                              fontsize=14,
                              frameon=True,
                              fancybox=True,
                              shadow=True)
        
        # Clean up the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        
        # Adjust layout with PROPER PADDING for research poster
        plt.tight_layout()
        # Adjust bottom padding - LESS space needed since no x-axis label
        bottom_padding = 0.18 if PLOT_SETTINGS["show_legend"] else 0.15  
        plt.subplots_adjust(bottom=bottom_padding, left=0.08, right=0.95, top=0.9)
        
        # Save with proper margins
        plot_path = output_dir / f'mep_clean_{hemisphere}_hemisphere.png'
        save_kwargs = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.3}
        if legend:
            save_kwargs['bbox_extra_artists'] = [legend]
        
        plt.savefig(plot_path, dpi=300, **save_kwargs)
        plt.show()
        
        figures.append(fig)
        print(f"✅ Clean {hemisphere} hemisphere plot saved: {plot_path}")
    
    return figures

def create_combined_figure(filtered_experiments, selected_muscles, config, y_max, output_dir):
    """Create combined figure with enhanced y-axis and closer groups"""
    
    # Create figure with better proportions using config settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_SETTINGS["figure_width"] * 1.25, PLOT_SETTINGS["figure_height"]))
    
    fig.suptitle('MEP Analysis - Hemisphere Comparison', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    n_experiments = len(filtered_experiments)
    n_muscles = len(selected_muscles)
    
    # CLOSER positioning with config setting
    muscle_base_positions = np.arange(n_muscles) * PLOT_SETTINGS["muscle_group_spacing"]  # Now uses config setting
    exp_width = 0.7  # Wider experiment spacing
    total_exp_width = (n_experiments - 1) * exp_width
    exp_offsets = np.linspace(-total_exp_width/2, total_exp_width/2, n_experiments)
    
    # Plot healthy data
    ax1.set_title('Healthy Hemisphere', fontsize=20, fontweight='bold', color='green', pad=20)
    plot_clean_hemisphere_data(ax1, filtered_experiments, selected_muscles, 'healthy', 
                              muscle_base_positions, exp_offsets, config, y_max)
    
    # Plot stroke data  
    ax2.set_title('Stroke Hemisphere', fontsize=20, fontweight='bold', color='red', pad=20)
    plot_clean_hemisphere_data(ax2, filtered_experiments, selected_muscles, 'stroke', 
                              muscle_base_positions, exp_offsets, config, y_max)
    
    # Configure both axes identically with ENHANCED Y-AXIS and CLOSER GROUPS
    for ax in [ax1, ax2]:
        total_width = muscle_base_positions[-1] if len(muscle_base_positions) > 0 else PLOT_SETTINGS["muscle_group_spacing"]
        margin_ratio = 0.25  # Increased padding
        left_margin = total_width * margin_ratio
        right_margin = total_width * margin_ratio
        
        ax.set_xlim(-left_margin, total_width + right_margin)
        
        # Apply ENHANCED Y-AXIS formatting
        nice_y_max = set_detailed_y_axis(ax, y_max * 1.1, 
                                        PLOT_SETTINGS["y_axis_minor_ticks"], 
                                        PLOT_SETTINGS["y_axis_major_ticks"])
        
        ax.set_xticks(muscle_base_positions)
        ax.set_xticklabels(selected_muscles, fontsize=16, fontweight='bold')
        ax.set_xlabel('Muscle Groups', fontsize=18, fontweight='bold', labelpad=20)
        ax.set_ylabel('MEP Amplitude (µV)', fontsize=18, fontweight='bold', labelpad=15)
        
        # Add subtle separators
        for i in range(1, len(muscle_base_positions)):
            separator_x = (muscle_base_positions[i-1] + muscle_base_positions[i]) / 2
            ax.axvline(x=separator_x, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
    
    # CONDITIONALLY create shared legend based on config
    legend = None
    if PLOT_SETTINGS["show_legend"]:
        legend_elements = []
        for exp_name, exp_data in filtered_experiments.items():
            config_exp = exp_data['config']
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=config_exp['color'], markersize=12, alpha=0.9,
                          label=config_exp['short_name'], markeredgecolor='white', markeredgewidth=1)
            )
        
        legend = fig.legend(handles=legend_elements, 
                           loc='upper center', 
                           bbox_to_anchor=(0.5, 0.02),
                           ncol=len(legend_elements),
                           fontsize=14,
                           frameon=True,
                           fancybox=True,
                           shadow=True)
    
    plt.tight_layout()
    # Adjust padding based on whether legend is shown
    bottom_padding = 0.15 if PLOT_SETTINGS["show_legend"] else 0.1
    plt.subplots_adjust(bottom=bottom_padding, top=0.9, left=0.08, right=0.95)
    
    # Save plot
    plot_path = output_dir / 'mep_clean_combined.png'
    save_kwargs = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.3}
    if legend:
        save_kwargs['bbox_extra_artists'] = [legend]
    
    plt.savefig(plot_path, dpi=300, **save_kwargs)
    plt.show()
    
    print(f"✅ Clean combined plot saved: {plot_path}")
    return fig

def calculate_detailed_mep_statistics(muscle_data: dict) -> dict:
    """Calculate comprehensive MEP statistics including raw and cleaned data
    
    ALWAYS calculates both raw and clean statistics regardless of config settings.
    The mean represents the arithmetic mean (average) of MEP amplitudes in µV.
    """
    
    detailed_stats = {}
    
    for muscle, data in muscle_data.items():
        healthy_raw = data['healthy_raw']
        stroke_raw = data['stroke_raw']
        healthy_clean = data['healthy']
        stroke_clean = data['stroke']
        
        if len(healthy_raw) == 0 and len(stroke_raw) == 0:
            continue
        
        muscle_stats = {
            'healthy_raw': {},
            'stroke_raw': {},
            'healthy_clean': {},
            'stroke_clean': {},
            'comparison_raw': {},
            'comparison_clean': {}
        }
        
        # RAW DATA STATISTICS (with outliers) - ALWAYS CALCULATED
        if len(healthy_raw) > 0:
            muscle_stats['healthy_raw'] = {
                'n': len(healthy_raw),
                'mean': healthy_raw.mean(),  # Arithmetic mean of MEP amplitudes (µV)
                'std': healthy_raw.std(),
                'sem': healthy_raw.std() / np.sqrt(len(healthy_raw)),
                'median': healthy_raw.median(),
                'q25': healthy_raw.quantile(0.25),
                'q75': healthy_raw.quantile(0.75),
                'min': healthy_raw.min(),
                'max': healthy_raw.max(),
                'cv': (healthy_raw.std() / healthy_raw.mean() * 100) if healthy_raw.mean() > 0 else 0
            }
        
        if len(stroke_raw) > 0:
            muscle_stats['stroke_raw'] = {
                'n': len(stroke_raw),
                'mean': stroke_raw.mean(),  # Arithmetic mean of MEP amplitudes (µV)
                'std': stroke_raw.std(),
                'sem': stroke_raw.std() / np.sqrt(len(stroke_raw)),
                'median': stroke_raw.median(),
                'q25': stroke_raw.quantile(0.25),
                'q75': stroke_raw.quantile(0.75),
                'min': stroke_raw.min(),
                'max': stroke_raw.max(),
                'cv': (stroke_raw.std() / stroke_raw.mean() * 100) if stroke_raw.mean() > 0 else 0
            }
        
        # CLEAN DATA STATISTICS (outliers removed) - ALWAYS CALCULATED
        if len(healthy_clean) > 0:
            muscle_stats['healthy_clean'] = {
                'n': len(healthy_clean),
                'mean': healthy_clean.mean(),  # Arithmetic mean of MEP amplitudes (µV) after outlier removal
                'std': healthy_clean.std(),
                'sem': healthy_clean.std() / np.sqrt(len(healthy_clean)),
                'median': healthy_clean.median(),
                'q25': healthy_clean.quantile(0.25),
                'q75': healthy_clean.quantile(0.75),
                'min': healthy_clean.min(),
                'max': healthy_clean.max(),
                'cv': (healthy_clean.std() / healthy_clean.mean() * 100) if healthy_clean.mean() > 0 else 0
            }
        
        if len(stroke_clean) > 0:
            muscle_stats['stroke_clean'] = {
                'n': len(stroke_clean),
                'mean': stroke_clean.mean(),  # Arithmetic mean of MEP amplitudes (µV) after outlier removal
                'std': stroke_clean.std(),
                'sem': stroke_clean.std() / np.sqrt(len(stroke_clean)),
                'median': stroke_clean.median(),
                'q25': stroke_clean.quantile(0.25),
                'q75': stroke_clean.quantile(0.75),
                'min': stroke_clean.min(),
                'max': stroke_clean.max(),
                'cv': (stroke_clean.std() / stroke_clean.mean() * 100) if stroke_clean.mean() > 0 else 0
            }
        
        # COMPARISON STATISTICS - ALWAYS CALCULATED FOR BOTH RAW AND CLEAN
        # Raw data comparison
        if len(healthy_raw) > 0 and len(stroke_raw) > 0:
            try:
                stat_raw, p_raw = stats.mannwhitneyu(healthy_raw, stroke_raw, alternative='two-sided')
                pooled_std_raw = np.sqrt(((len(healthy_raw) - 1) * np.var(healthy_raw, ddof=1) + 
                                        (len(stroke_raw) - 1) * np.var(stroke_raw, ddof=1)) / 
                                       (len(healthy_raw) + len(stroke_raw) - 2))
                cohens_d_raw = (healthy_raw.mean() - stroke_raw.mean()) / pooled_std_raw if pooled_std_raw > 0 else 0
                impairment_raw = (1 - stroke_raw.mean() / healthy_raw.mean()) * 100 if healthy_raw.mean() > 0 else 0
            except:
                stat_raw, p_raw, cohens_d_raw, impairment_raw = 0, 1.0, 0, 0
            
            muscle_stats['comparison_raw'] = {
                'p_value': p_raw,
                'cohens_d': cohens_d_raw,
                'impairment_percent': impairment_raw,
                'significant': p_raw < 0.05,
                'effect_size': 'large' if abs(cohens_d_raw) > 0.8 else 'medium' if abs(cohens_d_raw) > 0.5 else 'small'
            }
        
        # Clean data comparison - ALWAYS CALCULATED
        if len(healthy_clean) > 0 and len(stroke_clean) > 0:
            try:
                stat_clean, p_clean = stats.mannwhitneyu(healthy_clean, stroke_clean, alternative='two-sided')
                pooled_std_clean = np.sqrt(((len(healthy_clean) - 1) * np.var(healthy_clean, ddof=1) + 
                                          (len(stroke_clean) - 1) * np.var(stroke_clean, ddof=1)) / 
                                         (len(healthy_clean) + len(stroke_clean) - 2))
                cohens_d_clean = (healthy_clean.mean() - stroke_clean.mean()) / pooled_std_clean if pooled_std_clean > 0 else 0
                impairment_clean = (1 - stroke_clean.mean() / healthy_clean.mean()) * 100 if healthy_clean.mean() > 0 else 0
            except:
                stat_clean, p_clean, cohens_d_clean, impairment_clean = 0, 1.0, 0, 0
            
            muscle_stats['comparison_clean'] = {
                'p_value': p_clean,
                'cohens_d': cohens_d_clean,
                'impairment_percent': impairment_clean,
                'significant': p_clean < 0.05,
                'effect_size': 'large' if abs(cohens_d_clean) > 0.8 else 'medium' if abs(cohens_d_clean) > 0.5 else 'small'
            }
        
        detailed_stats[muscle] = muscle_stats
    
    return detailed_stats

def save_statistics_to_files(all_experiments: dict, output_dir: Path):
    """Save comprehensive statistics to CSV and text files with detailed MEP analysis"""
    
    if not (PLOT_SETTINGS["save_stats_csv"] or PLOT_SETTINGS["save_stats_txt"]):
        return
    
    print("\n📄 SAVING ENHANCED STATISTICS TO FILES")
    print("=" * 45)
    
    # Prepare data for different CSV files
    summary_csv_data = []
    detailed_csv_data = []
    detailed_text = []
    
    # Header for detailed text
    detailed_text.append("MEP ANALYSIS COMPREHENSIVE REPORT")
    detailed_text.append("=" * 60)
    detailed_text.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    detailed_text.append(f"Outlier removal: {PLOT_SETTINGS['remove_outliers']}")
    if PLOT_SETTINGS['remove_outliers']:
        detailed_text.append(f"Outlier method: {PLOT_SETTINGS['outlier_method']}")
        detailed_text.append(f"Upper limit: {PLOT_SETTINGS['outlier_upper_limit']} µV")
    detailed_text.append("")
    
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    for exp_name, exp_data in all_experiments.items():
        config = exp_data['config']
        
        # Calculate detailed statistics
        detailed_stats = calculate_detailed_mep_statistics(exp_data['muscle_data'])
        
        detailed_text.append(f"🐒 MONKEY: {config['short_name']} ({config['description']})")
        detailed_text.append("=" * 60)
        
        for muscle in muscle_names:
            if muscle in detailed_stats:
                stats = detailed_stats[muscle]
                
                detailed_text.append(f"\n💪 {muscle.upper()} MUSCLE:")
                detailed_text.append("-" * 40)
                
                # RAW DATA (with outliers)
                if 'healthy_raw' in stats and stats['healthy_raw']:
                    hr = stats['healthy_raw']
                    detailed_text.append(f"  📊 HEALTHY HEMISPHERE (Raw Data):")
                    detailed_text.append(f"     Mean: {hr['mean']:.2f} ± {hr['sem']:.2f} µV")
                    detailed_text.append(f"     Median: {hr['median']:.2f} µV")
                    detailed_text.append(f"     Range: {hr['min']:.1f} - {hr['max']:.1f} µV")
                    detailed_text.append(f"     IQR: {hr['q25']:.2f} - {hr['q75']:.2f} µV")
                    detailed_text.append(f"     CV: {hr['cv']:.1f}%")
                    detailed_text.append(f"     Sample size: {hr['n']}")
                
                if 'stroke_raw' in stats and stats['stroke_raw']:
                    sr = stats['stroke_raw']
                    detailed_text.append(f"  🩺 STROKE HEMISPHERE (Raw Data):")
                    detailed_text.append(f"     Mean: {sr['mean']:.2f} ± {sr['sem']:.2f} µV")
                    detailed_text.append(f"     Median: {sr['median']:.2f} µV")
                    detailed_text.append(f"     Range: {sr['min']:.1f} - {sr['max']:.1f} µV")
                    detailed_text.append(f"     IQR: {sr['q25']:.2f} - {sr['q75']:.2f} µV")
                    detailed_text.append(f"     CV: {sr['cv']:.1f}%")
                    detailed_text.append(f"     Sample size: {sr['n']}")
                
                # CLEAN DATA (outliers removed)
                if PLOT_SETTINGS['remove_outliers']:
                    detailed_text.append(f"\n  🧹 AFTER OUTLIER REMOVAL:")
                    
                    if 'healthy_clean' in stats and stats['healthy_clean']:
                        hc = stats['healthy_clean']
                        detailed_text.append(f"     Healthy Mean: {hc['mean']:.2f} ± {hc['sem']:.2f} µV (n={hc['n']})")
                        
                    if 'stroke_clean' in stats and stats['stroke_clean']:
                        sc = stats['stroke_clean']
                        detailed_text.append(f"     Stroke Mean: {sc['mean']:.2f} ± {sc['sem']:.2f} µV (n={sc['n']})")
                
                # STATISTICAL COMPARISONS
                if 'comparison_raw' in stats and stats['comparison_raw']:
                    cr = stats['comparison_raw']
                    sig_text = "***" if cr['significant'] else "ns"
                    detailed_text.append(f"\n  📈 STATISTICAL COMPARISON (Raw Data):")
                    detailed_text.append(f"     Impairment: {cr['impairment_percent']:.1f}%")
                    detailed_text.append(f"     Effect size: d={cr['cohens_d']:.3f} ({cr['effect_size']})")
                    detailed_text.append(f"     Significance: {sig_text} (p={cr['p_value']:.4f})")
                
                if PLOT_SETTINGS['remove_outliers'] and 'comparison_clean' in stats and stats['comparison_clean']:
                    cc = stats['comparison_clean']
                    sig_text = "***" if cc['significant'] else "ns"
                    detailed_text.append(f"  📈 STATISTICAL COMPARISON (Clean Data):")
                    detailed_text.append(f"     Impairment: {cc['impairment_percent']:.1f}%")
                    detailed_text.append(f"     Effect size: d={cc['cohens_d']:.3f} ({cc['effect_size']})")
                    detailed_text.append(f"     Significance: {sig_text} (p={cc['p_value']:.4f})")
                
                detailed_text.append("")
                
                # Add to CSV data
                # Summary CSV (one row per monkey-muscle combination)
                summary_row = {
                    'Monkey': config['short_name'],
                    'Muscle': muscle,
                    'Healthy_Mean_Raw': stats.get('healthy_raw', {}).get('mean', ''),
                    'Healthy_SEM_Raw': stats.get('healthy_raw', {}).get('sem', ''),
                    'Healthy_N_Raw': stats.get('healthy_raw', {}).get('n', ''),
                    'Stroke_Mean_Raw': stats.get('stroke_raw', {}).get('mean', ''),
                    'Stroke_SEM_Raw': stats.get('stroke_raw', {}).get('sem', ''),
                    'Stroke_N_Raw': stats.get('stroke_raw', {}).get('n', ''),
                    'Impairment_Pct_Raw': stats.get('comparison_raw', {}).get('impairment_percent', ''),
                    'P_Value_Raw': stats.get('comparison_raw', {}).get('p_value', ''),
                    'Cohens_D_Raw': stats.get('comparison_raw', {}).get('cohens_d', ''),
                    'Effect_Size_Raw': stats.get('comparison_raw', {}).get('effect_size', ''),
                    'Significant_Raw': stats.get('comparison_raw', {}).get('significant', ''),
                }
                
                # Add clean data - ALWAYS INCLUDE regardless of config
                summary_row.update({
                    'Healthy_Mean_Clean': stats.get('healthy_clean', {}).get('mean', ''),
                    'Healthy_SEM_Clean': stats.get('healthy_clean', {}).get('sem', ''),
                    'Healthy_N_Clean': stats.get('healthy_clean', {}).get('n', ''),
                    'Stroke_Mean_Clean': stats.get('stroke_clean', {}).get('mean', ''),
                    'Stroke_SEM_Clean': stats.get('stroke_clean', {}).get('sem', ''),
                    'Stroke_N_Clean': stats.get('stroke_clean', {}).get('n', ''),
                    'Impairment_Pct_Clean': stats.get('comparison_clean', {}).get('impairment_percent', ''),
                    'P_Value_Clean': stats.get('comparison_clean', {}).get('p_value', ''),
                    'Cohens_D_Clean': stats.get('comparison_clean', {}).get('cohens_d', ''),
                    'Effect_Size_Clean': stats.get('comparison_clean', {}).get('effect_size', ''),
                    'Significant_Clean': stats.get('comparison_clean', {}).get('significant', ''),
                })
                
                summary_csv_data.append(summary_row)
                
                # Detailed CSV (separate rows for healthy/stroke, raw/clean)
                for condition in ['healthy', 'stroke']:
                    for data_type in ['raw', 'clean']:
                        key = f"{condition}_{data_type}"
                        if key in stats and stats[key]:
                            detailed_row = {
                                'Monkey': config['short_name'],
                                'Muscle': muscle,
                                'Hemisphere': condition.capitalize(),
                                'Data_Type': data_type.capitalize(),
                                'N': stats[key]['n'],
                                'Mean': stats[key]['mean'],
                                'SEM': stats[key]['sem'],
                                'STD': stats[key]['std'],
                                'Median': stats[key]['median'],
                                'Q25': stats[key]['q25'],
                                'Q75': stats[key]['q75'],
                                'Min': stats[key]['min'],
                                'Max': stats[key]['max'],
                                'CV_Percent': stats[key]['cv']
                            }
                            detailed_csv_data.append(detailed_row)
        
        detailed_text.append("\n")
    
    # Overall summary across all monkeys
    detailed_text.append("🎯 OVERALL FINDINGS ACROSS ALL MONKEYS")
    detailed_text.append("=" * 50)
    
    for muscle in muscle_names:
        detailed_text.append(f"\n{muscle}:")
        
        # Collect data for overall analysis
        all_healthy_means_raw = []
        all_stroke_means_raw = []
        all_impairments_raw = []
        all_effect_sizes_raw = []
        all_p_values_raw = []
        
        if PLOT_SETTINGS['remove_outliers']:
            all_healthy_means_clean = []
            all_stroke_means_clean = []
            all_impairments_clean = []
            all_effect_sizes_clean = []
            all_p_values_clean = []
        
        for exp_data in all_experiments.values():
            detailed_stats = calculate_detailed_mep_statistics(exp_data['muscle_data'])
            if muscle in detailed_stats:
                stats = detailed_stats[muscle]
                
                if 'healthy_raw' in stats and 'stroke_raw' in stats:
                    all_healthy_means_raw.append(stats['healthy_raw']['mean'])
                    all_stroke_means_raw.append(stats['stroke_raw']['mean'])
                if 'comparison_raw' in stats:
                    all_impairments_raw.append(stats['comparison_raw']['impairment_percent'])
                    all_effect_sizes_raw.append(abs(stats['comparison_raw']['cohens_d']))
                    all_p_values_raw.append(stats['comparison_raw']['p_value'])
                
                if PLOT_SETTINGS['remove_outliers']:
                    if 'healthy_clean' in stats and 'stroke_clean' in stats:
                        all_healthy_means_clean.append(stats['healthy_clean']['mean'])
                        all_stroke_means_clean.append(stats['stroke_clean']['mean'])
                    if 'comparison_clean' in stats:
                        all_impairments_clean.append(stats['comparison_clean']['impairment_percent'])
                        all_effect_sizes_clean.append(abs(stats['comparison_clean']['cohens_d']))
                        all_p_values_clean.append(stats['comparison_clean']['p_value'])
        
        # Raw data summary
        if all_healthy_means_raw:
            detailed_text.append(f"  📊 Raw Data Summary:")
            detailed_text.append(f"     Average Healthy MEP: {np.mean(all_healthy_means_raw):.2f} ± {np.std(all_healthy_means_raw):.2f} µV")
            detailed_text.append(f"     Average Stroke MEP: {np.mean(all_stroke_means_raw):.2f} ± {np.std(all_stroke_means_raw):.2f} µV")
            detailed_text.append(f"     Average Impairment: {np.mean(all_impairments_raw):.1f} ± {np.std(all_impairments_raw):.1f}%")
            detailed_text.append(f"     Average Effect Size: d={np.mean(all_effect_sizes_raw):.3f}")
            significant_count_raw = sum(1 for p in all_p_values_raw if p < 0.05)
            detailed_text.append(f"     Significant Results: {significant_count_raw}/{len(all_p_values_raw)} monkeys")
        
        # Clean data summary - ALWAYS SHOW regardless of config
        if all_healthy_means_clean:
            detailed_text.append(f"  🧹 Clean Data Summary (outliers removed):")
            detailed_text.append(f"     Average Healthy MEP: {np.mean(all_healthy_means_clean):.2f} ± {np.std(all_healthy_means_clean):.2f} µV")
            detailed_text.append(f"     Average Stroke MEP: {np.mean(all_stroke_means_clean):.2f} ± {np.std(all_stroke_means_clean):.2f} µV")
            detailed_text.append(f"     Average Impairment: {np.mean(all_impairments_clean):.1f} ± {np.std(all_impairments_clean):.1f}%")
            detailed_text.append(f"     Average Effect Size: d={np.mean(all_effect_sizes_clean):.3f}")
            significant_count_clean = sum(1 for p in all_p_values_clean if p < 0.05)
            detailed_text.append(f"     Significant Results: {significant_count_clean}/{len(all_p_values_clean)} monkeys")
        
        detailed_text.append("")
    
    # Save CSV files
    if PLOT_SETTINGS["save_stats_csv"]:
        if summary_csv_data:
            summary_df = pd.DataFrame(summary_csv_data)
            summary_path = output_dir / 'mep_summary_statistics.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"📊 Summary CSV saved: {summary_path}")
        
        if detailed_csv_data:
            detailed_df = pd.DataFrame(detailed_csv_data)
            detailed_path = output_dir / 'mep_detailed_statistics.csv'
            detailed_df.to_csv(detailed_path, index=False)
            print(f"📊 Detailed CSV saved: {detailed_path}")
    
    # Save detailed text file
    if PLOT_SETTINGS["save_stats_txt"]:
        txt_path = output_dir / 'mep_comprehensive_report.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(detailed_text))
        print(f"📄 Comprehensive report saved: {txt_path}")

def print_comprehensive_summary(all_experiments: dict):
    """Print enhanced comprehensive summary with detailed MEP statistics"""
    
    print(f"\n" + "="*80)
    print("                    COMPREHENSIVE MEP ANALYSIS SUMMARY")
    print("="*80)
    
    muscle_names = [muscle for muscle, show in MUSCLE_CONFIG.items() if show]
    
    for exp_name, exp_data in all_experiments.items():
        config = exp_data['config']
        
        # Calculate detailed statistics
        detailed_stats = calculate_detailed_mep_statistics(exp_data['muscle_data'])
        
        print(f"\n🐒 MONKEY: {config['short_name']} ({config['description']})")
        print("-" * 70)
        
        for muscle in muscle_names:
            if muscle in detailed_stats:
                stats = detailed_stats[muscle]
                
                print(f"\n💪 {muscle}:")
                
                # Raw data (with outliers)
                if 'healthy_raw' in stats and 'stroke_raw' in stats and stats['healthy_raw'] and stats['stroke_raw']:
                    hr = stats['healthy_raw']
                    sr = stats['stroke_raw']
                    print(f"   📊 RAW DATA (all MEPs included):")
                    print(f"      Healthy: {hr['mean']:.1f} ± {hr['sem']:.1f} µV (n={hr['n']}, CV={hr['cv']:.1f}%) - arithmetic mean")
                    print(f"      Stroke:  {sr['mean']:.1f} ± {sr['sem']:.1f} µV (n={sr['n']}, CV={sr['cv']:.1f}%) - arithmetic mean")
                    
                    if 'comparison_raw' in stats and stats['comparison_raw']:
                        cr = stats['comparison_raw']
                        sig_text = "***" if cr['significant'] else "ns"
                        print(f"      Impairment: {cr['impairment_percent']:.1f}% | Effect: d={cr['cohens_d']:.2f} ({cr['effect_size']}) | p={cr['p_value']:.4f} {sig_text}")
                
                # Clean data (outliers removed)
                if PLOT_SETTINGS['remove_outliers'] and 'healthy_clean' in stats and 'stroke_clean' in stats and stats['healthy_clean'] and stats['stroke_clean']:
                    hc = stats['healthy_clean']
                    sc = stats['stroke_clean']
                    print(f"   🧹 CLEAN DATA (outliers removed):")
                    print(f"      Healthy: {hc['mean']:.1f} ± {hc['sem']:.1f} µV (n={hc['n']}, CV={hc['cv']:.1f}%)")
                    print(f"      Stroke:  {sc['mean']:.1f} ± {sc['sem']:.1f} µV (n={sc['n']}, CV={sc['cv']:.1f}%)")
                    
                    if 'comparison_clean' in stats and stats['comparison_clean']:
                        cc = stats['comparison_clean']
                        sig_text = "***" if cc['significant'] else "ns"
                        print(f"      Impairment: {cc['impairment_percent']:.1f}% | Effect: d={cc['cohens_d']:.2f} ({cc['effect_size']}) | p={cc['p_value']:.4f} {sig_text}")
    
    # Overall summary across all monkeys
    print(f"\n🎯 OVERALL FINDINGS ACROSS ALL MONKEYS:")
    print("-" * 50)
    
    for muscle in muscle_names:
        print(f"\n{muscle}:")
        
        # Collect data for overall analysis
        all_healthy_means_raw = []
        all_stroke_means_raw = []
        all_impairments_raw = []
        all_effect_sizes_raw = []
        all_p_values_raw = []
        
        if PLOT_SETTINGS['remove_outliers']:
            all_healthy_means_clean = []
            all_stroke_means_clean = []
            all_impairments_clean = []
            all_effect_sizes_clean = []
            all_p_values_clean = []
        
        for exp_data in all_experiments.values():
            detailed_stats = calculate_detailed_mep_statistics(exp_data['muscle_data'])
            if muscle in detailed_stats:
                stats = detailed_stats[muscle]
                
                if 'healthy_raw' in stats and 'stroke_raw' in stats and stats['healthy_raw'] and stats['stroke_raw']:
                    all_healthy_means_raw.append(stats['healthy_raw']['mean'])
                    all_stroke_means_raw.append(stats['stroke_raw']['mean'])
                if 'comparison_raw' in stats and stats['comparison_raw']:
                    all_impairments_raw.append(stats['comparison_raw']['impairment_percent'])
                    all_effect_sizes_raw.append(abs(stats['comparison_raw']['cohens_d']))
                    all_p_values_raw.append(stats['comparison_raw']['p_value'])
                
                if PLOT_SETTINGS['remove_outliers']:
                    if 'healthy_clean' in stats and 'stroke_clean' in stats and stats['healthy_clean'] and stats['stroke_clean']:
                        all_healthy_means_clean.append(stats['healthy_clean']['mean'])
                        all_stroke_means_clean.append(stats['stroke_clean']['mean'])
                    if 'comparison_clean' in stats and stats['comparison_clean']:
                        all_impairments_clean.append(stats['comparison_clean']['impairment_percent'])
                        all_effect_sizes_clean.append(abs(stats['comparison_clean']['cohens_d']))
                        all_p_values_clean.append(stats['comparison_clean']['p_value'])
        
        # Raw data summary
        if all_healthy_means_raw:
            print(f"   📊 Raw Data (n={len(all_healthy_means_raw)} monkeys):")
            print(f"      Healthy MEP: {np.mean(all_healthy_means_raw):.1f} ± {np.std(all_healthy_means_raw):.1f} µV")
            print(f"      Stroke MEP:  {np.mean(all_stroke_means_raw):.1f} ± {np.std(all_stroke_means_raw):.1f} µV")
            print(f"      Avg Impairment: {np.mean(all_impairments_raw):.1f} ± {np.std(all_impairments_raw):.1f}%")
            print(f"      Avg Effect Size: d={np.mean(all_effect_sizes_raw):.2f}")
            significant_count_raw = sum(1 for p in all_p_values_raw if p < 0.05)
            print(f"      Significant: {significant_count_raw}/{len(all_p_values_raw)} monkeys")
        
        # Clean data summary
        if PLOT_SETTINGS['remove_outliers'] and all_healthy_means_clean:
            print(f"   🧹 Clean Data (n={len(all_healthy_means_clean)} monkeys):")
            print(f"      Healthy MEP: {np.mean(all_healthy_means_clean):.1f} ± {np.std(all_healthy_means_clean):.1f} µV")
            print(f"      Stroke MEP:  {np.mean(all_stroke_means_clean):.1f} ± {np.std(all_stroke_means_clean):.1f} µV")
            print(f"      Avg Impairment: {np.mean(all_impairments_clean):.1f} ± {np.std(all_impairments_clean):.1f}%")
            print(f"      Avg Effect Size: d={np.mean(all_effect_sizes_clean):.2f}")
            significant_count_clean = sum(1 for p in all_p_values_clean if p < 0.05)
            print(f"      Significant: {significant_count_clean}/{len(all_p_values_clean)} monkeys")
    
    print(f"\n" + "="*80)
    print("📊 LEGEND: CV=Coefficient of Variation, d=Cohen's d, ***=p<0.05")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Enhanced MEP Analysis - Customizable Plots')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(), 
                       help='Base path containing experiment folders')
    parser.add_argument('--output-dir', type=Path, help='Output directory (optional)')
    parser.add_argument('--upper-limit', type=int, default=1000, help='Upper limit for MEP values (µV)')
    parser.add_argument('--method', type=str, default='iqr_conservative', 
                       choices=['iqr_conservative', 'physiological', 'percentile'],
                       help='Outlier removal method')
    
    # Plot control options
    parser.add_argument('--individual-only', action='store_true', 
                       help='Only create individual experiment plots')
    parser.add_argument('--comparison-only', action='store_true', 
                       help='Only create comparison plot')
    parser.add_argument('--combined-figure', action='store_true',
                       help='Create combined figure (healthy and stroke side-by-side)')
    
    args = parser.parse_args()
    
    # Set output directory
    if not args.output_dir:
        args.output_dir = args.base_path / 'enhanced_analysis_results'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("    ENHANCED MEP ANALYSIS - CUSTOMIZABLE PLOTS")
    print("="*60)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🧹 Cleaning method: {args.method}")
    print(f"📏 Upper limit: {args.upper_limit}µV")
    
    # Print current configuration
    print(f"\n🎯 CURRENT CONFIGURATION:")
    print(f"   Active monkeys: {[k for k, v in MONKEY_CONFIG.items() if v['show']]}")
    print(f"   Active muscles: {[k for k, v in MUSCLE_CONFIG.items() if v]}")
    print(f"   Show legend: {PLOT_SETTINGS['show_legend']}")
    print(f"   Remove outliers: {PLOT_SETTINGS['remove_outliers']}")
    print(f"   Y-axis ticks: Major={PLOT_SETTINGS['y_axis_major_ticks']}µV, Minor={PLOT_SETTINGS['y_axis_minor_ticks']}µV")
    print(f"   Muscle group spacing: {PLOT_SETTINGS['muscle_group_spacing']}")
    
    try:
        # Load all experiment data using config settings
        all_experiments = load_all_experiment_data(args.base_path)
        
        if not all_experiments:
            print("❌ No experiment data found")
            return
        
        print(f"\n✅ Successfully loaded {len(all_experiments)} experiments")
        
        # Create plots based on arguments
        if not args.comparison_only:
            create_individual_experiment_plots(all_experiments, args.output_dir)
        
        if not args.individual_only:
            # Determine plot config
            plot_config = {'separate_figures': not args.combined_figure}
            
            create_customizable_comparison_plot(
                all_experiments, args.output_dir,
                plot_config=plot_config
            )
        
        # Print comprehensive summary
        print_comprehensive_summary(all_experiments)
        
        # Save statistics to files if requested
        save_statistics_to_files(all_experiments, args.output_dir)
        
        print(f"\n📊 FILES CREATED IN: {args.output_dir}")
        if not args.comparison_only:
            print("   Individual experiment plots: *_individual_mep_analysis.png")
        if not args.individual_only:
            print("   Comparison plots: mep_clean_*_hemisphere.png")
        if PLOT_SETTINGS["save_stats_csv"]:
            print("   📊 Summary CSV: mep_summary_statistics.csv")
            print("   📊 Detailed CSV: mep_detailed_statistics.csv")
        if PLOT_SETTINGS["save_stats_txt"]:
            print("   📄 Comprehensive report: mep_comprehensive_report.txt")
        
        print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()