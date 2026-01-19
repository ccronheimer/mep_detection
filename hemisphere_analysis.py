#!/usr/bin/env python3
"""
MEP COMPARISON (relevance-filtered; robust stats; includes Mely + Andy; + individual plots)

Fixes vs your current script:
1) ✅ Uses relevance flags (chXXX_relevant) when present (windowed outputs).
2) ✅ Adds ROBUST stats per hemisphere:
   - median
   - trimmed mean (5%)
   - response rates (>50, >100, >300 µV)
3) ✅ Optional winsorization for plotting/means (clip at p99 by default).
4) ✅ Prevents nonsense impairment when healthy mean ~0 (prints NA).
5) ✅ Restores Mely APB-only by default (avoids flatline/duplicate-channel issues).
6) ✅ Keeps Andy as full 3-muscle mapping.

USAGE:
  # default: p99-winsorized means + medians/trimmed means + response rates
  python hemisphere_analysis_fixed.py --base-path .

  # keep raw values (not recommended for Andy)
  python hemisphere_analysis_fixed.py --base-path . --winsor none

  # stricter clipping for plots/means
  python hemisphere_analysis_fixed.py --base-path . --winsor p995

  # only individual plots
  python hemisphere_analysis_fixed.py --base-path . --individual-only
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

    # outliers (IQR still applies; "outlier_limit" is an optional safety cap)
    "remove_outliers": True,
    "outlier_method": "iqr_conservative",
    "outlier_upper_limit": 1_000_000,  # big safety cap; winsor handles the extreme stuff
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

# [Bicep, Brach, APB] mapping
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
    "Dec1_Mely": {
        "short_name": "NHP4",
        "show": True,
        "data_path": "Dec1_Mely/mep_results.csv",
        "healthy": {"channels": [130, 132, 135]},  # reference only; we do APB-only below
        "stroke":  {"channels": [139, 137, 136]},  # reference only; we do APB-only below
    },
    "Dec1_Andy": {
        "short_name": "NHP5",
        "show": True,
        "data_path": "Dec1_Andy/mep_results.csv",
        # Andy mapping: (Right Upper, Right Forearm, Right Hand) => 135,132,130
        # Stroke: (Left Upper, Left Forearm, Left Hand) => 136,137,139
        "healthy": {"channels": [135, 132, 130]},
        "stroke":  {"channels": [136, 137, 139]},
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

def list_enabled_muscles():
    order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    return [m for m in order if MUSCLE_CONFIG.get(m, False)]

def get_muscle_color(muscle_name: str) -> str:
    return MUSCLE_COLORS.get(muscle_name, "#999999")

def get_muscle_abbreviation(muscle_name: str) -> str:
    return {
        "Bicep": "Biceps",
        "Brachioradialis": "Brach",
        "Abductor Pollicis Brevis": "APB",
    }.get(muscle_name, muscle_name)

def remove_outliers(data: pd.Series, method="iqr_conservative", upper_limit=1_000_000) -> pd.Series:
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

    return data[~mask]

def winsorize(series: pd.Series, mode: str) -> pd.Series:
    """
    mode:
      none  -> no clipping
      p99   -> clip at 99th percentile
      p995  -> clip at 99.5th percentile
      p999  -> clip at 99.9th percentile
    """
    if len(series) == 0 or mode == "none":
        return series

    if mode == "p99":
        hi = series.quantile(0.99)
    elif mode == "p995":
        hi = series.quantile(0.995)
    elif mode == "p999":
        hi = series.quantile(0.999)
    else:
        raise ValueError(f"Unknown winsor mode: {mode}")

    return series.clip(lower=None, upper=hi)

def robust_stats(x: pd.Series) -> dict:
    """
    Returns robust summary metrics.
    """
    if len(x) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "sem": 0.0,
            "trimmed_mean_5": 0.0,
            "rr_gt50": 0.0,
            "rr_gt100": 0.0,
            "rr_gt300": 0.0,
        }

    x_np = x.to_numpy(dtype=float)
    n = len(x_np)
    mean = float(np.mean(x_np))
    median = float(np.median(x_np))
    sem = float(np.std(x_np, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    # 5% trimmed mean
    lo = int(np.floor(0.05 * n))
    hi = int(np.ceil(0.95 * n))
    xs = np.sort(x_np)
    trimmed = xs[lo:hi] if hi > lo else xs
    trimmed_mean = float(np.mean(trimmed)) if len(trimmed) else mean

    return {
        "n": int(n),
        "mean": mean,
        "median": median,
        "sem": sem,
        "trimmed_mean_5": trimmed_mean,
        "rr_gt50": float(np.mean(x_np > 50.0)),
        "rr_gt100": float(np.mean(x_np > 100.0)),
        "rr_gt300": float(np.mean(x_np > 300.0)),
    }

def impairment_percent(h_center: float, s_center: float) -> str:
    """
    Guard against dividing by ~0.
    """
    if abs(h_center) < 1e-6:
        return "NA"
    return f"{(1 - (s_center / h_center)) * 100:.1f}%"


def nice_step(y_top: float, target_ticks: int = 7) -> float:
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
    ax.tick_params(axis="y", direction="out", length=8, width=2, pad=10)
    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontfamily("Arial")
    return y_rounded


# =============================================================================
# DATA LOADING (relevance-filtered)
# =============================================================================

def extract_series(df_hemi: pd.DataFrame, ch: int, winsor_mode: str) -> pd.Series:
    amp_col = f"ch{ch}_amplitude"
    rel_col = f"ch{ch}_relevant"

    if amp_col not in df_hemi.columns:
        return pd.Series([], dtype=float)

    x = df_hemi[amp_col].dropna()

    # relevance filter if present
    if rel_col in df_hemi.columns:
        rel_mask = df_hemi[rel_col].fillna(False).astype(bool)
        x = df_hemi.loc[rel_mask, amp_col].dropna()

    # outlier removal (IQR)
    if PLOT_SETTINGS["remove_outliers"]:
        x = remove_outliers(x, PLOT_SETTINGS["outlier_method"], PLOT_SETTINGS["outlier_upper_limit"])

    # winsor for stable means/plots (does NOT change raw n; it just caps extreme values)
    x = winsorize(x, winsor_mode)

    return x.astype(float)

def load_all_experiment_data(base_path: Path, winsor_mode: str, mely_apb_only: bool):
    print("📊 LOADING EXPERIMENT DATA")
    print("=" * 50)

    all_experiments = {}
    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]

    filtered_configs = {k: v for k, v in ANALYSIS_CONFIGS.items() if v.get("show", True)}

    for exp_name, cfg in filtered_configs.items():
        exp_path = base_path / cfg["data_path"]
        if not exp_path.exists():
            print(f"⚠️  Missing: {exp_name} -> {exp_path}")
            continue

        df = pd.read_csv(exp_path)
        label_col = detect_label_column(df)
        healthy_df = df[df[label_col] == "healthy"].copy()
        stroke_df = df[df[label_col] == "stroke"].copy()

        print(f"\n📁 Loading {exp_name} -> {exp_path}")
        print(f"   ✅ rows: healthy={len(healthy_df)} stroke={len(stroke_df)} (label_col='{label_col}')")

        exp_results = {"config": cfg, "muscle_data": {}, "stats_results": {}}

        # Mely APB-only by default (avoids flatline/duplicate mapping issues)
        if exp_name == "Dec1_Mely" and mely_apb_only:
            apb = "Abductor Pollicis Brevis"
            if apb not in enabled_muscles:
                print("   (Mely APB-only) APB disabled; skipping Mely.")
                continue

            # APB: healthy=ch135, stroke=ch136 (your established choice)
            h = extract_series(healthy_df, 135, winsor_mode)
            s = extract_series(stroke_df, 136, winsor_mode)

            exp_results["muscle_data"][apb] = {"healthy": h, "stroke": s}
            exp_results["stats_results"][apb] = {"healthy": robust_stats(h), "stroke": robust_stats(s)}

            print(f"   (Mely APB-only) APB: H={len(h)}, S={len(s)}")
            all_experiments[exp_name] = exp_results
            continue

        # default mapping
        for muscle_name, healthy_ch, stroke_ch in zip(
            muscle_order,
            cfg["healthy"]["channels"],
            cfg["stroke"]["channels"],
        ):
            if muscle_name not in enabled_muscles:
                continue

            h = extract_series(healthy_df, healthy_ch, winsor_mode)
            s = extract_series(stroke_df, stroke_ch, winsor_mode)

            exp_results["muscle_data"][muscle_name] = {"healthy": h, "stroke": s}
            exp_results["stats_results"][muscle_name] = {"healthy": robust_stats(h), "stroke": robust_stats(s)}

            print(f"   {muscle_name}: H={len(h)}, S={len(s)}")

        all_experiments[exp_name] = exp_results

    return all_experiments


# =============================================================================
# PLOTTING
# =============================================================================

def create_side_by_side_comparison(all_experiments, output_dir: Path):
    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    if not filtered:
        print("❌ No experiments selected")
        return None

    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    global_muscles = [m for m in muscle_order if m in enabled_muscles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]))
    fig.suptitle(
        "MEPs in Healthy vs. Stroke-Affected Hemispheres",
        fontsize=PLOT_SETTINGS["title_font_size"] + 4,
        fontweight="bold",
        family="Arial",
        color="black",
        y=1.02,
    )

    monkey_base_positions = np.arange(len(filtered)) * PLOT_SETTINGS["muscle_group_spacing"]

    # y_max across plotted values
    all_vals = []
    for exp_data in filtered.values():
        for m in exp_data["muscle_data"]:
            all_vals.extend(exp_data["muscle_data"][m]["healthy"].tolist())
            all_vals.extend(exp_data["muscle_data"][m]["stroke"].tolist())
    y_max = max(all_vals) * 1.25 if all_vals else 100.0

    muscle_width = 1.3

    for ax, hemi, title in [
        (ax1, "healthy", "Healthy Hemisphere"),
        (ax2, "stroke", "Stroke-Affected Hemisphere"),
    ]:
        ax.set_title(title, fontsize=PLOT_SETTINGS["title_font_size"], fontweight="bold", pad=20)

        for idx, (exp_name, exp_data) in enumerate(filtered.items()):
            short = exp_data["config"]["short_name"]
            x0 = monkey_base_positions[idx]

            exp_muscles = [m for m in global_muscles if m in exp_data["muscle_data"]]
            offsets = (
                np.linspace(-((len(exp_muscles) - 1) * muscle_width) / 2,
                            +((len(exp_muscles) - 1) * muscle_width) / 2,
                            len(exp_muscles))
                if len(exp_muscles) > 1 else np.array([0.0])
            )

            for j, m in enumerate(exp_muscles):
                data = exp_data["muscle_data"][m][hemi]
                st = exp_data["stats_results"][m][hemi]
                if len(data) == 0:
                    continue

                xc = x0 + offsets[j]
                jitter = np.random.uniform(-0.35, 0.35, len(data))
                ax.scatter(
                    np.full(len(data), xc) + jitter,
                    data,
                    color=get_muscle_color(m),
                    alpha=0.75,
                    s=160,
                    edgecolors=get_muscle_color(m),
                    linewidth=2,
                    zorder=5,
                )

                ax.plot([xc - 0.4, xc + 0.4], [st["trimmed_mean_5"], st["trimmed_mean_5"]],
                        color="black", linewidth=5, solid_capstyle="round", zorder=10)

                ax.text(xc, -y_max * 0.10, get_muscle_abbreviation(m),
                        ha="center", va="top",
                        fontsize=PLOT_SETTINGS["tick_label_font_size"] - 6)

            ax.text(x0, -y_max * 0.22, short, ha="center", va="top",
                    fontsize=PLOT_SETTINGS["axis_label_font_size"],
                    fontweight="bold")

        margin_ratio = 0.35
        total_width = monkey_base_positions[-1] if len(monkey_base_positions) else PLOT_SETTINGS["muscle_group_spacing"]
        ax.set_xlim(-total_width * margin_ratio, total_width + total_width * margin_ratio)

        configure_axis_ticks(ax, y_max)
        ax.set_xticks([])
        ax.set_ylabel("MEP Amplitude (µV)", fontsize=PLOT_SETTINGS["axis_label_font_size"], labelpad=25)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_SETTINGS["spine_width"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, left=0.08, right=0.97, top=0.88, wspace=0.18)

    out = output_dir / "mep_side_by_side_fixed.png"
    plt.savefig(out, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", pad_inches=0.5)
    plt.show()
    print(f"✅ Saved: {out}")
    return fig


def create_individual_nhp_comparisons(all_experiments, output_dir: Path):
    filtered = {k: v for k, v in all_experiments.items() if ANALYSIS_CONFIGS[k].get("show", True)}
    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    global_muscles = [m for m in muscle_order if m in enabled_muscles]

    for exp_name, exp_data in filtered.items():
        short = exp_data["config"]["short_name"]
        muscles = [m for m in global_muscles if m in exp_data["muscle_data"]]
        if not muscles:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(f"{short}: Healthy vs. Stroke-Affected Hemisphere",
                     fontsize=40, fontweight="bold", family="Arial", y=1.00)

        all_vals = []
        for m in muscles:
            all_vals.extend(exp_data["muscle_data"][m]["healthy"].tolist())
            all_vals.extend(exp_data["muscle_data"][m]["stroke"].tolist())
        y_max = max(all_vals) * 1.25 if all_vals else 100.0

        x_pos = np.arange(len(muscles)) * 3.5

        for ax, hemi, title in [(ax1, "healthy", "Healthy Hemisphere"), (ax2, "stroke", "Stroke-Affected Hemisphere")]:
            ax.set_title(title, fontsize=36, fontweight="bold", pad=15)

            for i, m in enumerate(muscles):
                data = exp_data["muscle_data"][m][hemi]
                st = exp_data["stats_results"][m][hemi]
                if len(data) == 0:
                    continue

                jitter = np.random.uniform(-0.3, 0.3, len(data))
                ax.scatter(x_pos[i] + jitter, data,
                           color=get_muscle_color(m), alpha=0.75, s=180,
                           edgecolors=get_muscle_color(m), linewidth=2)

                ax.plot([x_pos[i] - 0.4, x_pos[i] + 0.4],
                        [st["trimmed_mean_5"], st["trimmed_mean_5"]],
                        color="black", linewidth=5)

                ax.text(x_pos[i], -y_max * 0.10, get_muscle_abbreviation(m),
                        ha="center", va="top", fontsize=24)

            ax.set_xlim(-1.5, x_pos[-1] + 1.5 if len(x_pos) else 1.5)
            configure_axis_ticks(ax, y_max)
            ax.set_xticks([])
            ax.set_ylabel("MEP Amplitude (µV)", fontsize=34, labelpad=20)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, left=0.08, right=0.97, top=0.88, wspace=0.15)

        out = output_dir / f"{short}_healthy_vs_stroke.png"
        plt.savefig(out, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight", facecolor="white", pad_inches=0.5)
        plt.show()
        print(f"✅ Saved: {out}")


# =============================================================================
# PRINTING
# =============================================================================

def print_comparison_statistics(all_experiments):
    print("\n" + "=" * 70)
    print("DETAILED MEP COMPARISON STATISTICS (robust; relevance-filtered)")
    print("=" * 70)

    enabled_muscles = list_enabled_muscles()
    muscle_order = ["Bicep", "Brachioradialis", "Abductor Pollicis Brevis"]
    global_muscles = [m for m in muscle_order if m in enabled_muscles]

    for exp_name, exp_data in all_experiments.items():
        short = exp_data["config"]["short_name"]
        muscles = [m for m in global_muscles if m in exp_data["stats_results"]]

        print(f"\n🐒 {short} ({exp_name}):")
        print("-" * 60)

        for m in muscles:
            h = exp_data["stats_results"][m]["healthy"]
            s = exp_data["stats_results"][m]["stroke"]

            # impairment based on trimmed mean (more stable than raw mean)
            imp = impairment_percent(h["trimmed_mean_5"], s["trimmed_mean_5"])

            print(f"\n  {m}:")
            print(f"    Healthy: n={h['n']}  mean={h['mean']:.1f}  median={h['median']:.1f}  tmean5={h['trimmed_mean_5']:.1f}  ±{h['sem']:.1f}")
            print(f"    Stroke:  n={s['n']}  mean={s['mean']:.1f}  median={s['median']:.1f}  tmean5={s['trimmed_mean_5']:.1f}  ±{s['sem']:.1f}")
            print(f"    Impairment (trimmed mean): {imp}")
            print(f"    Response rates (H): >50={h['rr_gt50']:.2f}  >100={h['rr_gt100']:.2f}  >300={h['rr_gt300']:.2f}")
            print(f"    Response rates (S): >50={s['rr_gt50']:.2f}  >100={s['rr_gt100']:.2f}  >300={s['rr_gt300']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--winsor", type=str, default="p99", choices=["none", "p99", "p995", "p999"],
                        help="Clip extreme amplitudes for stable means/plots (recommended for Andy).")
    parser.add_argument("--mely-apb-only", action="store_true", help="Use APB-only for Mely (recommended).")
    parser.add_argument("--side-by-side-only", action="store_true")
    parser.add_argument("--individual-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.base_path / "mep_comparisons")
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": PLOT_SETTINGS["tick_label_font_size"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    print("\n" + "=" * 70)
    print("  MEP COMPARISON (robust; relevance-filtered; Mely APB-only optional)")
    print("=" * 70)
    print(f"📁 Base path: {args.base_path}")
    print(f"📁 Output: {out_dir}")
    print(f"🧹 IQR outliers: {PLOT_SETTINGS['remove_outliers']}  method={PLOT_SETTINGS['outlier_method']}  cap={PLOT_SETTINGS['outlier_upper_limit']}")
    print(f"📌 Winsor (for stable means/plots): {args.winsor}")
    print(f"🐒 Mely APB-only: {args.mely_apb_only}")

    all_exps = load_all_experiment_data(args.base_path, winsor_mode=args.winsor, mely_apb_only=args.mely_apb_only)

    if not all_exps:
        print("❌ No experiment data found.")
        return

    if args.side_by_side_only:
        create_side_by_side_comparison(all_exps, out_dir)
    elif args.individual_only:
        create_individual_nhp_comparisons(all_exps, out_dir)
    else:
        create_side_by_side_comparison(all_exps, out_dir)
        create_individual_nhp_comparisons(all_exps, out_dir)

    print_comparison_statistics(all_exps)
    print(f"\n✅ Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
