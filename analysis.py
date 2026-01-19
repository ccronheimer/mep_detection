#!/usr/bin/env python3
"""
MEP Analysis Script (Poster Optimized) + Backward-Compatible Phase/Hemisphere Support
+ Individual-per-NHP plots

What this version fixes / adds:
1) ✅ Works with BOTH detection outputs:
   - old CSVs: column "hemisphere"
   - new windowed CSVs: column "phase"
   (Auto-detects which exists.)

2) ✅ Adds Dec1_Mely into ANALYSIS_CONFIGS + MONKEY_CONFIG.

3) ✅ Adds "individual shot" figures:
   - One plot PER NHP/experiment (stroke-only points + mean lines), same styling as group plot.

Notes:
- If you want Mely in NHPUES plots/correlation, set its NHPUES score in NHPUES_SCORES.
- Directory layout assumed:
    ./Nov5_Olive/mep_results.csv
    ./Nov5_Cheddar/mep_results.csv
    ./Oct31_Chive/mep_results.csv
    ./Dec1_Mely/mep_results.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# 🎯 POSTER-OPTIMIZED CONFIGURATION
# =============================================================================

PLOT_SETTINGS = {
    "figure_width": 25,
    "figure_height": 12,
    "title_font_size": 38,
    "axis_label_font_size": 38,
    "tick_label_font_size": 30,
    "legend_font_size": 28,
    "muscle_group_spacing": 4,

    "y_tick_interval": 10,
    "use_major_minor_ticks": False,
    "y_axis_max": None,

    "spine_width": 2,
    "background_color": "white",
    "dpi": 300,

    "remove_outliers": True,
    "outlier_method": "iqr_conservative",
    "outlier_upper_limit": 1000,
}

# Which muscles to include
MUSCLE_CONFIG = {
    "Bicep": True,
    "Brachioradialis": True,
    "Abductor Pollicis Brevis": True,
}

MUSCLE_COLORS = {
    "Bicep": "#FFB3BA",
    "Brachioradialis": "#FF7A92",
    "Abductor Pollicis Brevis": "#7BC8D9",
}

# NHPUES scores (update Mely’s score!)
NHPUES_SCORES = {
    "NHP1": 10.3846154,  # Olive
    "NHP2": 11.3076923,  # Cheddar
    "NHP3": 15.2307692,  # Chive
    "NHP4": 0.0,         # TODO: set Mely’s true score
    "NHP5": 0.0,         # TODO: set Andy’s true score
}

# =============================================================================
# MONKEY CONFIG
# - name: must match NHPUES_SCORES keys if you want NHPUES plots
# =============================================================================

MONKEY_CONFIG = {
    "Nov5_Olive": {
        "show": True,
        "color": "#E31A1C",
        "name": "NHP1",
        "mean_line_color": "black",
        "mean_line_width": 6,
        "mean_line_length": 0.45,
    },
    "Nov5_Cheddar": {
        "show": True,
        "color": "#33A02C",
        "name": "NHP2",
        "mean_line_color": "black",
        "mean_line_width": 6,
        "mean_line_length": 0.45,
    },
    "Oct31_Chive": {
        "show": True,
        "color": "#FF7F00",
        "name": "NHP3",
        "mean_line_color": "black",
        "mean_line_width": 6,
        "mean_line_length": 0.45,
    },

    # ✅ ADD: Dec1_Mely
    "Dec1_Mely": {
        "show": True,
        "color": "#2E2E2E",
        "name": "NHP4",
        "mean_line_color": "black",
        "mean_line_width": 6,
        "mean_line_length": 0.45,
    },
     # ✅ ADD: Dec1_Andy
    "Dec1_Andy": {
        "show": True,
        "color": "#2E2E2E",
        "name": "NHP5",
        "mean_line_color": "black",
        "mean_line_width": 6,
        "mean_line_length": 0.45,
    },

    # (Optional / keep as you had)
    "Nov5_Chive": {"show": False, "color": "#1F78B4", "name": "NHP3", "mean_line_color": "black", "mean_line_width": 6, "mean_line_length": 0.45},
    "Oct31_Cheddar": {"show": False, "color": "#6A3D9A", "name": "NHP2", "mean_line_color": "black", "mean_line_width": 6, "mean_line_length": 0.45},
}

# =============================================================================
# ANALYSIS CONFIGS
# - IMPORTANT: channels must match the amplitude columns in mep_results.csv
# =============================================================================

ANALYSIS_CONFIGS = {
    "Nov5_Olive": {
        "description": "Nov5 Olive Experiment",
        "short_name": "Olive",
        "color": "#E31A1C",
        "show": True,
        "data_path": "Nov5_Olive/mep_results.csv",
        "healthy": {
            "channels": [2, 4, 7],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 182,
        },
        "stroke": {
            "channels": [8, 9, 11],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 263,
        },
    },
    "Nov5_Cheddar": {
        "description": "Nov5 Cheddar Experiment",
        "short_name": "Cheddar",
        "color": "#33A02C",
        "show": True,
        "data_path": "Nov5_Cheddar/mep_results.csv",
        "healthy": {
            "channels": [2, 4, 12],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 105,
        },
        "stroke": {
            "channels": [14, 9, 13],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 195,
        },
    },
    "Oct31_Chive": {
        "description": "Oct31 Chive Experiment",
        "short_name": "Chive",
        "color": "#FF7F00",
        "show": True,
        "data_path": "Oct31_Chive/mep_results.csv",
        "healthy": {
            "channels": [1, 2, 3],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 85,
        },
        "stroke": {
            "channels": [4, 5, 6],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 205,
        },
    },

    # ✅ ADD: Dec1_Mely (your channel IDs from chan*.mat)
    "Dec1_Mely": {
        "description": "Dec 1 Mely Experiment",
        "short_name": "Mely",
        "color": "#2E2E2E",
        "show": True,
        "data_path": "Dec1_Mely/mep_results.csv",
        "healthy": {
            "channels": [130, 132, 135],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 120,
        },
        "stroke": {
            "channels": [139, 137, 136],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 122,
        },
    },

    "Dec1_Andy": {
        "description": "Dec 1 Andy Experiment",
        "short_name": "Andy",
        "color": "#1F78B4",  # (optional) or keep "#2E2E2E" if you want
        "show": True,
        "data_path": "Dec1_Andy/mep_results.csv",
        "healthy": {
            # Andy mapping: Channel 7/4/2 -> 135/132/130
            "channels": [135, 132, 130],
            "channel_names": ["Right Upper", "Right Forearm", "Right Hand"],
            "expected_pulses": 164,
        },
        "stroke": {
            # Andy mapping: Channel 8/9/11 -> 136/137/139
            "channels": [136, 137, 139],
            "channel_names": ["Left Upper", "Left Forearm", "Left Hand"],
            "expected_pulses": 121,
        },   
    }
}

# Apply MONKEY_CONFIG overrides (show/color/labeling)
for exp_name, monkey_settings in MONKEY_CONFIG.items():
    if exp_name in ANALYSIS_CONFIGS:
        ANALYSIS_CONFIGS[exp_name]["short_name"] = monkey_settings["name"]  # NHP1/NHP2/...
        ANALYSIS_CONFIGS[exp_name]["color"] = monkey_settings["color"]
        ANALYSIS_CONFIGS[exp_name]["show"] = monkey_settings["show"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_color"] = monkey_settings["mean_line_color"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_width"] = monkey_settings["mean_line_width"]
        ANALYSIS_CONFIGS[exp_name]["mean_line_length"] = monkey_settings["mean_line_length"]

# Matplotlib style
plt.style.use("default")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": PLOT_SETTINGS["tick_label_font_size"],
    "axes.titlesize": PLOT_SETTINGS["title_font_size"],
    "axes.labelsize": PLOT_SETTINGS["axis_label_font_size"],
    "xtick.labelsize": PLOT_SETTINGS["tick_label_font_size"],
    "ytick.labelsize": PLOT_SETTINGS["tick_label_font_size"],
    "legend.fontsize": PLOT_SETTINGS["legend_font_size"],
    "figure.facecolor": PLOT_SETTINGS["background_color"],
    "axes.facecolor": PLOT_SETTINGS["background_color"],
    "savefig.facecolor": PLOT_SETTINGS["background_color"],
    "savefig.dpi": PLOT_SETTINGS["dpi"],
})

# =============================================================================
# UTILS
# =============================================================================

def get_muscle_color(muscle_name: str) -> str:
    return MUSCLE_COLORS.get(muscle_name, "#999999")

def map_nhp_to_scale_score(nhp_id: str) -> float:
    return float(NHPUES_SCORES.get(nhp_id, 0.0))

def configure_axis_ticks(ax, y_max, fixed_max=None):
    """
    If fixed_max is provided, use it.
    Otherwise auto-scale to y_max with a sensible tick interval.
    """
    if fixed_max is not None:
        y_top = float(fixed_max)
    else:
        y_top = float(y_max)

    # Choose tick step based on range
    if y_top <= 120:
        step = 10
    elif y_top <= 300:
        step = 25
    elif y_top <= 700:
        step = 50
    else:
        step = 100

    y_max_rounded = float(np.ceil(y_top / step) * step)

    ax.set_ylim(0, y_max_rounded)
    ax.set_yticks(np.arange(0, y_max_rounded + step, step))

    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", direction="in", length=8, width=2)

    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontfamily("Arial")

    return y_max_rounded


def remove_extreme_outliers(data: pd.Series, method="iqr_conservative", upper_limit=1000):
    if len(data) == 0:
        return data, pd.Series([], dtype=bool), {}

    original_count = len(data)

    if method == "iqr_conservative":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = min(q3 + 3.0 * iqr, upper_limit)
        mask = (data < lower) | (data > upper)
    elif method == "physiological":
        mask = (data < 0) | (data > upper_limit)
    elif method == "percentile":
        lower = data.quantile(0.01)
        upper = min(data.quantile(0.99), upper_limit)
        mask = (data < lower) | (data > upper)
    else:
        mask = pd.Series([False] * len(data), index=data.index)

    clean = data[~mask]
    removed = original_count - len(clean)

    stats_info = {
        "original_count": original_count,
        "removed_count": removed,
        "percent_removed": (removed / original_count * 100) if original_count else 0,
        "method": method,
        "upper_limit": upper_limit,
        "final_range": (float(clean.min()), float(clean.max())) if len(clean) else (0.0, 0.0),
    }
    return clean, mask, stats_info

def detect_label_column(df: pd.DataFrame) -> str:
    """Backward compatible: old files use 'hemisphere', new windowed use 'phase'."""
    if "hemisphere" in df.columns:
        return "hemisphere"
    if "phase" in df.columns:
        return "phase"
    raise ValueError("CSV missing required label column: expected 'hemisphere' or 'phase'")

# =============================================================================
# DATA LOADING + STATS
# =============================================================================

def calculate_experiment_statistics(muscle_data: dict) -> dict:
    out = {}
    for muscle, d in muscle_data.items():
        healthy = d["healthy"]
        stroke = d["stroke"]
        if len(healthy) == 0 or len(stroke) == 0:
            continue

        healthy_stats = {
            "mean": float(healthy.mean()),
            "std": float(healthy.std()),
            "sem": float(healthy.std() / np.sqrt(len(healthy))),
            "median": float(healthy.median()),
            "n": int(len(healthy)),
        }
        stroke_stats = {
            "mean": float(stroke.mean()),
            "std": float(stroke.std()),
            "sem": float(stroke.std() / np.sqrt(len(stroke))),
            "median": float(stroke.median()),
            "n": int(len(stroke)),
        }

        try:
            _, p = stats.mannwhitneyu(healthy, stroke, alternative="two-sided")
        except Exception:
            p = 1.0

        mean_imp = (1 - stroke_stats["mean"] / healthy_stats["mean"]) * 100 if healthy_stats["mean"] > 0 else 0.0

        pooled = np.sqrt(
            ((len(healthy) - 1) * np.var(healthy, ddof=1) + (len(stroke) - 1) * np.var(stroke, ddof=1))
            / (len(healthy) + len(stroke) - 2)
        )
        d_eff = (healthy_stats["mean"] - stroke_stats["mean"]) / pooled if pooled > 0 else 0.0

        out[muscle] = {
            "healthy": healthy_stats,
            "stroke": stroke_stats,
            "p_value": float(p),
            "mean_impairment_percent": float(mean_imp),
            "cohens_d": float(d_eff),
            "significant": bool(p < 0.05),
            "healthy_name": d["healthy_name"],
            "stroke_name": d["stroke_name"],
        }

    return out

def load_all_experiment_data(base_path: Path):
    print("📊 LOADING EXPERIMENT DATA")
    print("=" * 50)

    muscle_names = [m for m, show in MUSCLE_CONFIG.items() if show]
    filtered_configs = {k: v for k, v in ANALYSIS_CONFIGS.items() if v.get("show", True)}

    print(f"🐒 Active experiments: {list(filtered_configs.keys())}")
    print(f"💪 Active muscles: {muscle_names}")
    print(f"🧹 Remove outliers: {PLOT_SETTINGS['remove_outliers']} ({PLOT_SETTINGS['outlier_method']})")

    all_exps = {}

    for exp_name, cfg in filtered_configs.items():
        exp_path = base_path / cfg["data_path"]
        if not exp_path.exists():
            print(f"⚠️ Missing: {exp_name} at {exp_path}")
            continue

        print(f"📁 Loading {exp_name} -> {exp_path}")
        df = pd.read_csv(exp_path)
        label_col = detect_label_column(df)

        healthy_df = df[df[label_col] == "healthy"].copy()
        stroke_df = df[df[label_col] == "stroke"].copy()
        print(f"   ✅ {len(healthy_df)} healthy + {len(stroke_df)} stroke rows")

        exp_results = {
            "config": cfg,
            "muscle_data": {},
            "cleaning_report": {},
            "stats_results": {},
        }

        # Fixed muscle order for the 3 channels
        muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]

        for i, (healthy_ch, stroke_ch, muscle_name) in enumerate(
            zip(cfg["healthy"]["channels"], cfg["stroke"]["channels"], muscle_order)
        ):
            if muscle_name not in muscle_names:
                continue

            healthy_col = f"ch{healthy_ch}_amplitude"
            stroke_col = f"ch{stroke_ch}_amplitude"

            healthy_raw = healthy_df[healthy_col].dropna() if healthy_col in healthy_df.columns else pd.Series([], dtype=float)
            stroke_raw = stroke_df[stroke_col].dropna() if stroke_col in stroke_df.columns else pd.Series([], dtype=float)

            if PLOT_SETTINGS["remove_outliers"]:
                healthy_clean, _, h_stats = remove_extreme_outliers(
                    healthy_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"]
                )
                stroke_clean, _, s_stats = remove_extreme_outliers(
                    stroke_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"]
                )
            else:
                healthy_clean, stroke_clean = healthy_raw.copy(), stroke_raw.copy()
                h_stats = {"original_count": len(healthy_raw), "removed_count": 0, "percent_removed": 0}
                s_stats = {"original_count": len(stroke_raw), "removed_count": 0, "percent_removed": 0}

            exp_results["muscle_data"][muscle_name] = {
                "healthy": healthy_clean,
                "stroke": stroke_clean,
                "healthy_raw": healthy_raw,
                "stroke_raw": stroke_raw,
                "healthy_channel": healthy_ch,
                "stroke_channel": stroke_ch,
                "healthy_name": cfg["healthy"]["channel_names"][i],
                "stroke_name": cfg["stroke"]["channel_names"][i],
            }

            exp_results["cleaning_report"][muscle_name] = {"healthy": h_stats, "stroke": s_stats}
            print(f"   {muscle_name}: H={len(healthy_clean)}/{len(healthy_raw)}, S={len(stroke_clean)}/{len(stroke_raw)}")

        exp_results["stats_results"] = calculate_experiment_statistics(exp_results["muscle_data"])
        all_exps[exp_name] = exp_results

    return all_exps

# =============================================================================
# PLOTS
# =============================================================================

def create_group_stroke_points_plot(all_experiments: dict, output_dir: Path):
    print("\n📊 GROUP: STROKE PLOT (POINTS + MEAN)")
    print("=" * 55)

    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    muscle_names = [m for m, show in MUSCLE_CONFIG.items() if show]
    if not filtered:
        print("❌ No experiments selected")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
    ax.set_title("MEPs in Stroke-Affected Hemispheres", fontsize=PLOT_SETTINGS["title_font_size"], fontweight="bold", pad=40)

    n_exp = len(filtered)
    n_muscle = len(muscle_names)

    monkey_base_positions = np.arange(n_exp) * PLOT_SETTINGS["muscle_group_spacing"]
    muscle_width = 1.2
    total_muscle_width = (n_muscle - 1) * muscle_width
    muscle_offsets = np.linspace(-total_muscle_width / 2, total_muscle_width / 2, n_muscle)

    # y-limit
    all_stroke_vals = []
    for exp in filtered.values():
        for m in muscle_names:
            if m in exp["muscle_data"]:
                all_stroke_vals.extend(exp["muscle_data"][m]["stroke"].tolist())
    y_max = max(all_stroke_vals) * 1.2 if all_stroke_vals else 100.0

    # plot
    for monkey_idx, (exp_name, exp_data) in enumerate(filtered.items()):
        cfg = exp_data["config"]
        nhp_label = cfg["short_name"]
        monkey_x = monkey_base_positions[monkey_idx]

        for muscle_idx, muscle in enumerate(muscle_names):
            if muscle not in exp_data["muscle_data"] or muscle not in exp_data["stats_results"]:
                continue

            stroke_data = exp_data["muscle_data"][muscle]["stroke"]
            stroke_stats = exp_data["stats_results"][muscle]["stroke"]
            color = get_muscle_color(muscle)

            if len(stroke_data) == 0:
                continue

            x_center = monkey_x + muscle_offsets[muscle_idx]
            jitter = np.random.uniform(-0.4, 0.4, len(stroke_data))
            x_coords = np.full(len(stroke_data), x_center) + jitter

            ax.scatter(x_coords, stroke_data, color=color, alpha=1.0, s=200, marker="o",
                       edgecolors=color, linewidth=2, zorder=5)

            mean_val = stroke_stats["mean"]
            mlw = cfg.get("mean_line_length", 0.45)
            ax.plot([x_center - mlw, x_center + mlw], [mean_val, mean_val],
                    color=cfg.get("mean_line_color", "black"),
                    linewidth=cfg.get("mean_line_width", 6),
                    solid_capstyle="round", zorder=10)

            muscle_label = "APB" if muscle == "Abductor Pollicis Brevis" else muscle
            ax.text(x_center, -y_max * 0.06, muscle_label, ha="center", va="top",
                    fontsize=PLOT_SETTINGS["tick_label_font_size"], color="black")

        ax.text(monkey_x, -y_max * 0.15, nhp_label, ha="center", va="top",
                fontsize=PLOT_SETTINGS["axis_label_font_size"], fontweight="bold", color="black")

    # axes formatting
    margin_ratio = 0.25
    total_width = monkey_base_positions[-1] if len(monkey_base_positions) else PLOT_SETTINGS["muscle_group_spacing"]
    ax.set_xlim(-total_width * margin_ratio, total_width + total_width * margin_ratio)

    configure_axis_ticks(ax, y_max)
    ax.set_xticks([])
    ax.set_ylabel("MEP Amplitude (µV)", fontsize=PLOT_SETTINGS["axis_label_font_size"], labelpad=20)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SETTINGS["spine_width"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, left=0.1, right=0.95, top=0.88)

    out = output_dir / "poster_stroke_individual_points.png"
    plt.savefig(out, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.2)
    plt.show()
    print(f"✅ Saved: {out}")
    return fig

def create_individual_stroke_plot_per_nhp(all_experiments: dict, output_dir: Path):
    """
    One figure per experiment/NHP showing stroke-only points + mean lines per muscle.
    Fixes:
      - No negative-y text (prevents massive whitespace in saved PNG)
      - Auto y-axis scaling (prevents clipping for Mely)
      - Uses normal x-ticks for muscle labels
    """
    print("\n📸 INDIVIDUAL: STROKE PLOTS (ONE PER NHP)")
    print("=" * 55)

    muscle_names = [m for m, show in MUSCLE_CONFIG.items() if show]
    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    if not filtered:
        print("❌ No experiments selected")
        return

    # Pretty labels
    def muscle_label(m):
        return "APB" if m == "Abductor Pollicis Brevis" else m

    for exp_name, exp_data in filtered.items():
        cfg = exp_data["config"]
        nhp_label = cfg["short_name"]

        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
        ax.set_title(f"Stroke Hemisphere MEPs — {nhp_label}",
                     fontsize=PLOT_SETTINGS["title_font_size"], fontweight="bold", pad=30)

        # Gather stroke values for y scaling
        stroke_vals = []
        for m in muscle_names:
            if m in exp_data["muscle_data"]:
                stroke_vals.extend(exp_data["muscle_data"][m]["stroke"].tolist())

        if len(stroke_vals) == 0:
            print(f"⚠️ No stroke data for {exp_name}")
            plt.close(fig)
            continue

        # Auto y max with a bit of headroom
        y_max = float(np.nanmax(stroke_vals)) * 1.15

        # X positions
        x_centers = np.arange(len(muscle_names))

        for i, muscle in enumerate(muscle_names):
            if muscle not in exp_data["muscle_data"] or muscle not in exp_data["stats_results"]:
                continue

            stroke_data = exp_data["muscle_data"][muscle]["stroke"]
            if len(stroke_data) == 0:
                continue

            stroke_stats = exp_data["stats_results"][muscle]["stroke"]
            color = get_muscle_color(muscle)

            x_center = x_centers[i]
            jitter = np.random.uniform(-0.18, 0.18, len(stroke_data))
            x_coords = np.full(len(stroke_data), x_center) + jitter

            ax.scatter(
                x_coords, stroke_data,
                color=color, alpha=1.0, s=220, marker="o",
                edgecolors=color, linewidth=2, zorder=5
            )

            # Mean line
            mean_val = stroke_stats["mean"]
            half_len = 0.22
            ax.plot(
                [x_center - half_len, x_center + half_len],
                [mean_val, mean_val],
                color=cfg.get("mean_line_color", "black"),
                linewidth=cfg.get("mean_line_width", 6),
                solid_capstyle="round", zorder=10
            )

        # Axes formatting
        ax.set_xlim(-0.6, len(muscle_names) - 1 + 0.6)

        # AUTO ticks for individual plots (do NOT force 110)
        configure_axis_ticks(ax, y_max, fixed_max=None)

        ax.set_ylabel("MEP Amplitude (µV)",
                      fontsize=PLOT_SETTINGS["axis_label_font_size"],
                      fontweight="normal",
                      labelpad=18)

        ax.set_xticks(x_centers)
        ax.set_xticklabels([muscle_label(m) for m in muscle_names], fontfamily="Arial")
        ax.tick_params(axis="x", length=0, labelsize=PLOT_SETTINGS["tick_label_font_size"])

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_width"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14, left=0.10, right=0.95, top=0.88)

        out = output_dir / f"stroke_points_{exp_name}.png"
        plt.savefig(out, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.2)
        plt.show()
        print(f"✅ Saved: {out}")


def create_mean_nhpues_bar_plot(all_experiments: dict, output_dir: Path):
    print("\n📊 NHPUES BAR PLOT")
    print("=" * 40)

    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    if not filtered:
        print("❌ No experiments selected")
        return None

    # One bar per NHP label
    nhps = []
    scores = []
    for exp_name, exp_data in filtered.items():
        nhp_label = exp_data["config"]["short_name"]
        if nhp_label not in nhps:
            nhps.append(nhp_label)
            scores.append(map_nhp_to_scale_score(nhp_label))

    # sort for stable order
    order = np.argsort(nhps)
    nhps = [nhps[i] for i in order]
    scores = [scores[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
    ax.set_title("Mean NHPUES Scores", fontsize=PLOT_SETTINGS["title_font_size"], fontweight="bold", pad=40)

    x = np.arange(len(nhps))
    bar_width = 0.36
    ax.bar(x, scores, width=bar_width, color="#7BC8D9", edgecolor="black", linewidth=2)

    y_max = max(scores) * 1.2 if scores else 20
    y_max_round = int(np.ceil(y_max / 2) * 2)
    ax.set_ylim(0, y_max_round)
    ax.set_yticks(np.arange(0, y_max_round + 2, 2))
    ax.tick_params(axis="y", direction="in", length=8, width=2)

    for i, name in enumerate(nhps):
        ax.text(i, -y_max_round * 0.08, name, ha="center", va="top",
                fontsize=PLOT_SETTINGS["axis_label_font_size"], fontweight="bold")

    ax.set_xticks([])
    ax.set_ylabel("NHPUES Score (0-25)", fontsize=PLOT_SETTINGS["axis_label_font_size"], labelpad=20)

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_SETTINGS["spine_width"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, left=0.12, right=0.95, top=0.88)

    out = output_dir / "mean_nhpues_bar_plot.png"
    plt.savefig(out, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.3)
    plt.show()
    print(f"✅ Saved: {out}")
    return fig

def print_summary(all_experiments: dict):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    muscle_names = [m for m, show in MUSCLE_CONFIG.items() if show]

    for exp_name, exp_data in all_experiments.items():
        cfg = exp_data["config"]
        print(f"\n{exp_name} ({cfg['short_name']}):")
        for muscle in muscle_names:
            if muscle not in exp_data["stats_results"]:
                continue
            s = exp_data["stats_results"][muscle]
            print(
                f"  {muscle:24s} "
                f"H mean={s['healthy']['mean']:.2f} (n={s['healthy']['n']}) | "
                f"S mean={s['stroke']['mean']:.2f} (n={s['stroke']['n']}) | "
                f"p={s['p_value']:.3g} | d={s['cohens_d']:.2f}"
            )

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MEP Analysis (Poster Optimized) + Individual NHP plots")
    parser.add_argument("--base-path", type=Path, default=Path.cwd(), help="Base path containing experiment folders")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (optional)")
    parser.add_argument("--stroke-only", action="store_true", help="Create group stroke plot only")
    parser.add_argument("--bar-only", action="store_true", help="Create NHPUES bar plot only")
    parser.add_argument("--individual-only", action="store_true", help="Create individual per-NHP stroke plots only")
    parser.add_argument("--all", action="store_true", help="Create all plots")

    args = parser.parse_args()
    out_dir = args.output_dir or (args.base_path / "poster_analysis_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MEP ANALYSIS")
    print("=" * 60)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output:    {out_dir}")

    all_experiments = load_all_experiment_data(args.base_path)
    if not all_experiments:
        print("❌ No experiment data found.")
        return

    # Decide what to create
    if args.stroke_only:
        create_group_stroke_points_plot(all_experiments, out_dir)

    elif args.bar_only:
        create_mean_nhpues_bar_plot(all_experiments, out_dir)

    elif args.individual_only:
        create_individual_stroke_plot_per_nhp(all_experiments, out_dir)

    elif args.all or (not args.stroke_only and not args.bar_only and not args.individual_only):
        # Default: all
        create_group_stroke_points_plot(all_experiments, out_dir)
        create_individual_stroke_plot_per_nhp(all_experiments, out_dir)
        create_mean_nhpues_bar_plot(all_experiments, out_dir)

    print_summary(all_experiments)
    print(f"\n✅ Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()