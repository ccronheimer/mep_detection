#!/usr/bin/env python3
"""
MEP Healthy vs Stroke-Affected Hemisphere Comparison

UPDATED:
- Adds Dec1_Mely (NHP4) and Dec1_Andy (NHP5)
- Uses APB-only for Mely:
    healthy APB = ch135
    stroke  APB = ch136
- Works with BOTH detection CSV formats:
    - old: "hemisphere"
    - new: "phase"
- Adds individual-only shots (one per NHP), like your other analysis script.

USAGE:
  # all (side-by-side + individual per NHP)
  python mep_compare_all.py --base-path .

  # only side-by-side
  python mep_compare_all.py --base-path . --side-by-side-only

  # only individual per-NHP plots
  python mep_compare_all.py --base-path . --individual-only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

MONKEY_CONFIG = {
    "Nov5_Olive":   {"show": True, "name": "NHP1", "color": "#FF69B4"},
    "Nov5_Cheddar": {"show": True, "name": "NHP2", "color": "#C71585"},
    "Oct31_Chive":  {"show": True, "name": "NHP3", "color": "#FF69B4"},
    "Dec1_Mely":    {"show": True, "name": "NHP4", "color": "#2E2E2E"},
    "Dec1_Andy":    {"show": True, "name": "NHP5", "color": "#1F78B4"},
}

PLOT_SETTINGS = {
    "figure_width": 32,
    "figure_height": 14,
    "title_font_size": 42,
    "axis_label_font_size": 36,
    "tick_label_font_size": 28,
    "muscle_group_spacing": 5,
    "spine_width": 3,
    "dpi": 300,

    # outliers
    "remove_outliers": True,
    "outlier_method": "iqr_conservative",
    "outlier_upper_limit": 1000,
}

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

# Channel mapping for each experiment:
# [Bicep, Brachioradialis, APB]
ANALYSIS_CONFIGS = {
    "Nov5_Olive": {
        "short_name": "NHP1",
        "show": True,
        "data_path": "Nov5_Olive/mep_results.csv",
        "healthy": {"channels": [2, 4, 7]},
        "stroke":  {"channels": [8, 9, 11]},
    },
    "Nov5_Cheddar": {
        "short_name": "NHP2",
        "show": True,
        "data_path": "Nov5_Cheddar/mep_results.csv",
        "healthy": {"channels": [2, 4, 12]},
        "stroke":  {"channels": [14, 9, 13]},
    },
    "Oct31_Chive": {
        "short_name": "NHP3",
        "show": True,
        "data_path": "Oct31_Chive/mep_results.csv",
        "healthy": {"channels": [1, 2, 3]},
        "stroke":  {"channels": [4, 5, 6]},
    },

    # Mely (special handling later: APB-only)
    "Dec1_Mely": {
        "short_name": "NHP4",
        "show": True,
        "data_path": "Dec1_Mely/mep_results.csv",
        "healthy": {"channels": [130, 132, 135]},  # reference mapping
        "stroke":  {"channels": [139, 137, 136]},  # reference mapping
    },

    # Andy
    "Dec1_Andy": {
        "short_name": "NHP5",
        "show": True,
        "data_path": "Dec1_Andy/mep_results.csv",
        "healthy": {"channels": [135, 132, 130]},  # Right upper, forearm, hand (converted)
        "stroke":  {"channels": [136, 137, 139]},  # Left upper, forearm, hand (converted)
    },
}

# Apply MONKEY_CONFIG overrides
for exp_name, monkey_settings in MONKEY_CONFIG.items():
    if exp_name in ANALYSIS_CONFIGS:
        ANALYSIS_CONFIGS[exp_name]["short_name"] = monkey_settings["name"]
        ANALYSIS_CONFIGS[exp_name]["show"] = monkey_settings["show"]


# =============================================================================
# UTILS
# =============================================================================

def detect_label_column(df: pd.DataFrame) -> str:
    if "hemisphere" in df.columns:
        return "hemisphere"
    if "phase" in df.columns:
        return "phase"
    raise ValueError("CSV missing label column: expected 'hemisphere' or 'phase'")

def remove_outliers(data: pd.Series, method="iqr_conservative", upper_limit=1000) -> pd.Series:
    if len(data) == 0:
        return data

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
        mask = pd.Series(False, index=data.index)

    filtered = data[~mask]
    removed = len(data) - len(filtered)
    if removed > 0 and len(filtered) > 0:
        print(f"      Removed {removed}/{len(data)} outliers (method={method}, limit={upper_limit} µV)")
    return filtered

def nice_step(y_top: float, target_ticks: int = 7) -> float:
    """
    Choose a 'nice' tick step (1,2,5)*10^k so we end up with ~target_ticks.
    """
    if y_top <= 0:
        return 1.0
    raw = y_top / max(3, target_ticks)
    exp = np.floor(np.log10(raw))
    base = raw / (10 ** exp)

    if base <= 1:
        m = 1
    elif base <= 2:
        m = 2
    elif base <= 5:
        m = 5
    else:
        m = 10

    return float(m * (10 ** exp))

def configure_axis_ticks(ax, y_max: float) -> float:
    y_top = float(max(1.0, y_max))
    step = nice_step(y_top, target_ticks=7)
    y_rounded = float(np.ceil(y_top / step) * step)

    ax.set_ylim(0, y_rounded)
    ax.set_yticks(np.arange(0, y_rounded + step, step))

    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(
        axis="y",
        direction="out",
        length=8,
        width=2,
        labelsize=PLOT_SETTINGS["tick_label_font_size"],
        pad=10,
    )

    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontfamily("Arial")

    return y_rounded

def get_muscle_color(muscle_name: str) -> str:
    return MUSCLE_COLORS.get(muscle_name, "#999999")

def get_muscle_abbreviation(muscle_name: str) -> str:
    return {
        "Bicep": "Biceps",
        "Brachioradialis": "Brach",
        "Abductor Pollicis Brevis": "APB",
    }.get(muscle_name, muscle_name)

def list_enabled_muscles():
    order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    return [m for m in order if MUSCLE_CONFIG.get(m, False)]


# =============================================================================
# DATA LOADING
# =============================================================================

def _stats(x: pd.Series) -> dict:
    if len(x) == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "n": 0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "sem": float(x.std() / np.sqrt(len(x))),
        "n": int(len(x)),
    }

def load_all_experiment_data(base_path: Path):
    print("📊 LOADING EXPERIMENT DATA")
    print("=" * 50)

    all_experiments = {}
    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]

    filtered_configs = {k: v for k, v in ANALYSIS_CONFIGS.items() if v.get("show", True)}

    for exp_name, config in filtered_configs.items():
        exp_path = base_path / config["data_path"]
        if not exp_path.exists():
            print(f"⚠️  {exp_name} data not found: {exp_path}")
            continue

        print(f"📁 Loading {exp_name}...")
        df = pd.read_csv(exp_path)
        label_col = detect_label_column(df)

        healthy_df = df[df[label_col] == "healthy"].copy()
        stroke_df = df[df[label_col] == "stroke"].copy()

        exp_results = {"config": config, "muscle_data": {}, "stats_results": {}}

        # # Special: Mely APB-only
        # if exp_name == "Dec1_Mely":
        #     apb = "Abductor Pollicis Brevis"
        #     if apb not in enabled_muscles:
        #         print("   (Mely APB-only) APB disabled; skipping.")
        #         continue

        #     healthy_ch = 135
        #     stroke_ch = 136
        #     hc = f"ch{healthy_ch}_amplitude"
        #     sc = f"ch{stroke_ch}_amplitude"

        #     h_raw = healthy_df[hc].dropna() if hc in healthy_df.columns else pd.Series([], dtype=float)
        #     s_raw = stroke_df[sc].dropna() if sc in stroke_df.columns else pd.Series([], dtype=float)

        #     if PLOT_SETTINGS["remove_outliers"]:
        #         h = remove_outliers(h_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"])
        #         s = remove_outliers(s_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"])
        #     else:
        #         h, s = h_raw.copy(), s_raw.copy()

        #     exp_results["muscle_data"][apb] = {"healthy": h, "stroke": s}
        #     exp_results["stats_results"][apb] = {"healthy": _stats(h), "stroke": _stats(s)}

        #     print(f"   (Mely APB-only) APB: H={len(h)}, S={len(s)}")
        #     all_experiments[exp_name] = exp_results
        #     continue

        # Default: all muscles per mapping
        for healthy_ch, stroke_ch, muscle_name in zip(
            config["healthy"]["channels"],
            config["stroke"]["channels"],
            muscle_order,
        ):
            if muscle_name not in enabled_muscles:
                continue

            hc = f"ch{healthy_ch}_amplitude"
            sc = f"ch{stroke_ch}_amplitude"

            h_raw = healthy_df[hc].dropna() if hc in healthy_df.columns else pd.Series([], dtype=float)
            s_raw = stroke_df[sc].dropna() if sc in stroke_df.columns else pd.Series([], dtype=float)

            if PLOT_SETTINGS["remove_outliers"]:
                h = remove_outliers(h_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"])
                s = remove_outliers(s_raw, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"])
            else:
                h, s = h_raw.copy(), s_raw.copy()

            exp_results["muscle_data"][muscle_name] = {"healthy": h, "stroke": s}
            exp_results["stats_results"][muscle_name] = {"healthy": _stats(h), "stroke": _stats(s)}

            print(f"   {muscle_name}: H={len(h)}, S={len(s)}")

        all_experiments[exp_name] = exp_results

    return all_experiments


# =============================================================================
# PLOTTING
# =============================================================================

def create_side_by_side_comparison(all_experiments, output_dir: Path):
    print("\n📊 CREATING SIDE-BY-SIDE COMPARISON")
    print("=" * 55)

    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    if not filtered:
        print("❌ No experiments selected")
        return None

    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    muscle_names = [m for m in muscle_order if m in enabled_muscles]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]),
    )

    fig.suptitle(
        "MEPs in Healthy vs. Stroke-Affected Hemispheres",
        fontsize=PLOT_SETTINGS["title_font_size"] + 4,
        fontweight="bold",
        family="Arial",
        color="black",
        y=1.02,
    )

    n_experiments = len(filtered)
    monkey_base_positions = np.arange(n_experiments) * PLOT_SETTINGS["muscle_group_spacing"]

    muscle_width = 1.3
    total_muscle_width = (len(muscle_names) - 1) * muscle_width if len(muscle_names) > 1 else 0
    muscle_offsets = (
        np.linspace(-total_muscle_width / 2, total_muscle_width / 2, len(muscle_names))
        if len(muscle_names) else np.array([0.0])
    )

    # y_max across all points
    all_vals = []
    for exp_data in filtered.values():
        for m in exp_data["muscle_data"].keys():
            all_vals.extend(exp_data["muscle_data"][m]["healthy"].tolist())
            all_vals.extend(exp_data["muscle_data"][m]["stroke"].tolist())
    y_max = max(all_vals) * 1.25 if all_vals else 100

    for ax, hemisphere, title in [
        (ax1, "healthy", "Healthy Hemisphere"),
        (ax2, "stroke", "Stroke-Affected Hemisphere"),
    ]:
        ax.set_title(
            title,
            fontsize=PLOT_SETTINGS["title_font_size"],
            fontweight="bold",
            color="black",
            family="Arial",
            pad=20,
        )

        for monkey_idx, (_, exp_data) in enumerate(filtered.items()):
            short_name = exp_data["config"]["short_name"]
            monkey_x = monkey_base_positions[monkey_idx]

            # only plot muscles that exist for this exp (Mely only has APB)
            exp_muscles = [m for m in muscle_names if m in exp_data["muscle_data"]]
            exp_offsets = (
                np.linspace(
                    -((len(exp_muscles) - 1) * muscle_width) / 2,
                    +((len(exp_muscles) - 1) * muscle_width) / 2,
                    len(exp_muscles),
                ) if len(exp_muscles) > 1 else np.array([0.0])
            )

            for i, muscle in enumerate(exp_muscles):
                data = exp_data["muscle_data"][muscle][hemisphere]
                stats_ = exp_data["stats_results"][muscle][hemisphere]
                color = get_muscle_color(muscle)

                if len(data) == 0:
                    continue

                x_center = monkey_x + exp_offsets[i]
                jitter = np.random.uniform(-0.35, 0.35, len(data))
                x_coords = np.full(len(data), x_center) + jitter

                ax.scatter(
                    x_coords, data,
                    color=color, alpha=0.8, s=180,
                    marker="o", edgecolors=color,
                    linewidth=2, zorder=5,
                )

                ax.plot(
                    [x_center - 0.4, x_center + 0.4],
                    [stats_["mean"], stats_["mean"]],
                    color="black", linewidth=5,
                    solid_capstyle="round", zorder=10,
                )

                ax.text(
                    x_center, -y_max * 0.10, get_muscle_abbreviation(muscle),
                    ha="center", va="top",
                    fontsize=PLOT_SETTINGS["tick_label_font_size"] - 6,
                    fontweight="normal", color="black",
                    fontfamily="Arial",
                )

            ax.text(
                monkey_x, -y_max * 0.22, short_name,
                ha="center", va="top",
                fontsize=PLOT_SETTINGS["axis_label_font_size"],
                fontweight="bold", color="black",
                fontfamily="Arial",
            )

        margin_ratio = 0.35
        total_width = monkey_base_positions[-1] if len(monkey_base_positions) else PLOT_SETTINGS["muscle_group_spacing"]
        ax.set_xlim(-total_width * margin_ratio, total_width + total_width * margin_ratio)

        configure_axis_ticks(ax, y_max)

        ax.set_xticks([])
        ax.set_ylabel(
            "MEP Amplitude (µV)",
            fontsize=PLOT_SETTINGS["axis_label_font_size"],
            fontweight="normal",
            color="black",
            fontfamily="Arial",
            labelpad=25,
        )

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_width"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, left=0.08, right=0.97, top=0.88, wspace=0.18)

    plot_path = output_dir / "mep_side_by_side_fixed.png"
    plt.savefig(plot_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", pad_inches=0.5)
    plt.show()
    print(f"✅ Saved: {plot_path}")
    return fig


def create_individual_nhp_comparisons(all_experiments, output_dir: Path):
    """
    One figure per NHP/experiment: Healthy vs Stroke panels.
    For Mely this will only show APB.
    """
    print("\n📸 INDIVIDUAL: HEALTHY vs STROKE (ONE PER NHP)")
    print("=" * 55)

    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    if not filtered:
        print("❌ No experiments selected")
        return

    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    global_muscles = [m for m in muscle_order if m in enabled_muscles]

    for exp_name, exp_data in filtered.items():
        short_name = exp_data["config"]["short_name"]

        muscles = [m for m in global_muscles if m in exp_data["muscle_data"]]
        if not muscles:
            print(f"⚠️  No muscles available for {exp_name}")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(
            f"{short_name}: Healthy vs. Stroke-Affected Hemisphere",
            fontsize=40,
            fontweight="bold",
            family="Arial",
            color="black",
            y=1.00,
        )

        # y max for this NHP
        all_vals = []
        for m in muscles:
            all_vals.extend(exp_data["muscle_data"][m]["healthy"].tolist())
            all_vals.extend(exp_data["muscle_data"][m]["stroke"].tolist())
        y_max = max(all_vals) * 1.25 if all_vals else 100

        muscle_positions = np.arange(len(muscles)) * 3.5

        for ax, hemisphere, title in [
            (ax1, "healthy", "Healthy Hemisphere"),
            (ax2, "stroke", "Stroke-Affected Hemisphere"),
        ]:
            ax.set_title(title, fontsize=36, fontweight="bold", color="black", family="Arial", pad=15)

            for i, muscle in enumerate(muscles):
                data = exp_data["muscle_data"][muscle][hemisphere]
                stats_ = exp_data["stats_results"][muscle][hemisphere]
                color = get_muscle_color(muscle)

                if len(data) == 0:
                    continue

                x_pos = muscle_positions[i]
                jitter = np.random.uniform(-0.3, 0.3, len(data))
                x_coords = x_pos + jitter

                ax.scatter(
                    x_coords, data,
                    color=color, alpha=0.8, s=200,
                    edgecolors=color, linewidth=2, zorder=5,
                )

                ax.plot(
                    [x_pos - 0.4, x_pos + 0.4],
                    [stats_["mean"], stats_["mean"]],
                    color="black", linewidth=5, zorder=10,
                )

                ax.text(
                    x_pos, -y_max * 0.10, get_muscle_abbreviation(muscle),
                    ha="center", va="top",
                    fontsize=24, fontweight="normal",
                    fontfamily="Arial", color="black",
                )

            ax.set_xlim(-1.5, muscle_positions[-1] + 1.5 if len(muscle_positions) else 1.5)
            configure_axis_ticks(ax, y_max)

            ax.set_xticks([])
            ax.set_ylabel(
                "MEP Amplitude (µV)",
                fontsize=34,
                fontweight="normal",
                fontfamily="Arial",
                labelpad=20,
                color="black",
            )

            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_edgecolor("black")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, left=0.08, right=0.97, top=0.88, wspace=0.15)

        plot_path = output_dir / f"{short_name}_healthy_vs_stroke.png"
        plt.savefig(plot_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", pad_inches=0.5)
        plt.show()
        print(f"✅ Saved: {plot_path}")


# =============================================================================
# STATS PRINTING
# =============================================================================

def print_comparison_statistics(all_experiments):
    print("\n" + "=" * 70)
    print("DETAILED MEP COMPARISON STATISTICS")
    print("=" * 70)

    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    global_muscles = [m for m in muscle_order if m in enabled_muscles]

    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}

    for exp_name, exp_data in filtered.items():
        short_name = exp_data["config"]["short_name"]
        muscles = [m for m in global_muscles if m in exp_data["stats_results"]]

        print(f"\n🐒 {short_name} ({exp_name}):")
        print("-" * 60)

        for muscle in muscles:
            h = exp_data["stats_results"][muscle]["healthy"]
            s = exp_data["stats_results"][muscle]["stroke"]
            impairment = (1 - s["mean"] / h["mean"]) * 100 if h["mean"] > 0 else 0

            print(f"\n  {muscle}:")
            print(f"    Healthy: {h['mean']:.1f} ± {h['sem']:.1f} µV (n={h['n']})")
            print(f"    Stroke:  {s['mean']:.1f} ± {s['sem']:.1f} µV (n={s['n']})")
            print(f"    Impairment: {impairment:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MEP Comparison (phase/hemisphere compatible; Mely APB-only; + individual plots)")
    parser.add_argument("--base-path", type=Path, default=Path.cwd(), help="Base path containing experiment folders")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--outlier-method", type=str, default="iqr_conservative",
                        choices=["iqr_conservative", "physiological", "percentile"])
    parser.add_argument("--outlier-limit", type=float, default=1000, help="Upper limit for outlier removal (µV)")
    parser.add_argument("--side-by-side-only", action="store_true", help="Generate only the side-by-side plot")
    parser.add_argument("--individual-only", action="store_true", help="Generate only individual per-NHP plots")

    args = parser.parse_args()

    PLOT_SETTINGS["outlier_method"] = args.outlier_method
    PLOT_SETTINGS["outlier_upper_limit"] = args.outlier_limit

    if not args.output_dir:
        args.output_dir = args.base_path / "mep_comparisons"
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Matplotlib config
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": PLOT_SETTINGS["tick_label_font_size"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    print("=" * 70)
    print("  MEP COMPARISON (Mely APB-only; includes Andy; + individual plots)")
    print("=" * 70)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🧹 Outlier method: {args.outlier_method}")
    print(f"🧹 Outlier limit: {args.outlier_limit} µV")

    all_experiments = load_all_experiment_data(args.base_path)
    if not all_experiments:
        print("❌ No experiment data found")
        return

    if args.side_by_side_only:
        create_side_by_side_comparison(all_experiments, args.output_dir)
    elif args.individual_only:
        create_individual_nhp_comparisons(all_experiments, args.output_dir)
    else:
        create_side_by_side_comparison(all_experiments, args.output_dir)
        create_individual_nhp_comparisons(all_experiments, args.output_dir)

    print_comparison_statistics(all_experiments)
    print(f"\n✅ Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
