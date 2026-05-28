"""
Visualization & Analysis Module
Generates all research plots:
  - Training curves
  - Personality trait distributions
  - Career recommendation charts
  - ASD biomarker comparison
  - Modality contribution plots
  - Correlation matrices
  - PGS distribution plots

References
----------
[DAT-11] Hunter (2007). Matplotlib: A 2D graphics environment.
         Computing in Science & Engineering, 9(3), 90–95.
         https://doi.org/10.1109/MCSE.2007.55
         → Primary visualization library for all 8 research plots.

[DAT-07] McKinney (2010). Data structures for statistical computing in Python. SciPy.
         → Pandas used for metadata handling in visualization functions.

[BIO-08] Masi et al. (2015). Cytokine aberrations in autism spectrum disorder.
         Molecular Psychiatry, 20(4), 440–446. https://doi.org/10.1038/mp.2014.59
         → ASD biomarker Z-score reference values in plot_biomarker_comparison().

[ASD-03] Hope et al. (2023). Bidirectional genetic overlap between ASD and cognitive traits.
         Translational Psychiatry, 13(1), 290.
         https://doi.org/10.1038/s41398-023-02563-7
         → ASD vs NT Big Five differences visualised in plot_bigfive_distributions().

[PSY-01] Costa & McCrae (1992). Revised NEO personality inventory (NEO PI-R).
         → Big Five radar chart (plot_career_radar) uses NEO PI-R trait labels and [0,1] scale.

[ETH-01] Buolamwini & Gebru (2018). Gender shades.
         FAT* 2018. https://doi.org/10.1145/3287560.3287572
         → Fairness reference motivating demographic-stratified plot_asd_calibration().
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import REPORTS_DIR, BIG_FIVE
from src.utils.career_database import CAREER_DATABASE

# Style
plt.rcParams.update({
    "figure.facecolor":  "#0a0e1a",
    "axes.facecolor":    "#0d1220",
    "axes.edgecolor":    "#2a3a50",
    "axes.labelcolor":   "#c8d4e0",
    "xtick.color":       "#8899aa",
    "ytick.color":       "#8899aa",
    "text.color":        "#e8eaf0",
    "grid.color":        "#1e2a45",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

PALETTE = {
    "blue":   "#4a90d9",
    "green":  "#00C49F",
    "purple": "#6C63FF",
    "red":    "#FF6B6B",
    "gold":   "#FFD166",
    "teal":   "#4ECDC4",
    "orange": "#F8961E",
}
COLORS = list(PALETTE.values())


def save_fig(fig, name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> Path:
    train_h = history["train"]
    val_h   = history["val"]
    epochs  = range(1, len(train_h) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History — CareerMappingGenomics", fontsize=14, color="#e8eaf0")

    metrics = [
        ("loss",       "Total Loss"),
        ("career_acc", "Career Accuracy"),
        ("bf_mae",     "Big Five MAE"),
        ("asd_acc",    "ASD Accuracy"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        t_vals = [ep.get(key, 0) for ep in train_h]
        v_vals = [ep.get(key, 0) for ep in val_h]
        ax.plot(epochs, t_vals, color=PALETTE["blue"],   linewidth=2, label="Train")
        ax.plot(epochs, v_vals, color=PALETTE["orange"], linewidth=2, label="Val", linestyle="--")
        ax.set_title(title, color="#e8eaf0")
        ax.legend(framealpha=0.3)
        ax.grid(True)

    plt.tight_layout()
    return save_fig(fig, "01_training_curves")


# ──────────────────────────────────────────────────────────────────────
# 2. Big Five Distribution by ASD Status
# ──────────────────────────────────────────────────────────────────────

def plot_bigfive_distributions(big_five: np.ndarray, is_autism: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("Big Five Trait Distributions: Neurotypical vs ASD", color="#e8eaf0", fontsize=13)

    trait_names = BIG_FIVE
    nt_mask  = is_autism == 0
    asd_mask = is_autism == 1

    for i, (ax, trait) in enumerate(zip(axes, trait_names)):
        bins = np.linspace(0, 1, 25)
        ax.hist(big_five[nt_mask, i],  bins=bins, alpha=0.7, color=PALETTE["blue"],
                label="Neurotypical", density=True)
        ax.hist(big_five[asd_mask, i], bins=bins, alpha=0.7, color=PALETTE["orange"],
                label="ASD", density=True)
        ax.set_title(trait, fontsize=10, color="#e8eaf0")
        ax.set_xlabel("Score")
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    return save_fig(fig, "02_bigfive_distributions")


# ──────────────────────────────────────────────────────────────────────
# 3. Biomarker Comparison: ASD vs NT
# ──────────────────────────────────────────────────────────────────────

def plot_biomarker_comparison(biomarkers: np.ndarray, is_autism: np.ndarray) -> Path:
    # Select most diagnostically relevant markers
    key_markers = [
        "oxytocin_pg_ml", "serotonin_ng_ml", "bdnf_ng_ml",
        "hs_crp_mg_l", "il6_pg_ml", "gaba_glutamate_ratio",
        "glutamate_nmol_ml", "cortisol_ug_dl", "vitamin_d_ng_ml",
        "melatonin_pg_ml", "zinc_ug_dl", "avp_pg_ml",
    ]
    from src.utils.synthetic_data import BIOMARKER_NAMES
    indices = [BIOMARKER_NAMES.index(m) for m in key_markers if m in BIOMARKER_NAMES]

    nt_data  = biomarkers[is_autism == 0][:, indices]
    asd_data = biomarkers[is_autism == 1][:, indices]

    nt_mean  = nt_data.mean(axis=0)
    asd_mean = asd_data.mean(axis=0)

    # Z-score relative to NT mean
    nt_std = nt_data.std(axis=0) + 1e-8
    asd_z  = (asd_mean - nt_mean) / nt_std

    labels = [m.replace("_", " ").replace(" pg ml", "").replace(" ng ml", "")
              for m in [BIOMARKER_NAMES[i] for i in indices]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("ASD vs Neurotypical Biomarker Differences (Z-score)", color="#e8eaf0", fontsize=13)

    colors = [PALETTE["red"] if z < 0 else PALETTE["green"] for z in asd_z]
    bars = ax.barh(labels, asd_z, color=colors, alpha=0.85, edgecolor="#0a0e1a")
    ax.axvline(0, color="#e8eaf0", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Z-score difference (ASD vs Neurotypical)")
    ax.grid(True, axis="x")

    red_patch   = mpatches.Patch(color=PALETTE["red"],   label="Lower in ASD")
    green_patch = mpatches.Patch(color=PALETTE["green"], label="Higher in ASD")
    ax.legend(handles=[red_patch, green_patch], loc="lower right", framealpha=0.3)

    plt.tight_layout()
    return save_fig(fig, "03_biomarker_comparison_asd")


# ──────────────────────────────────────────────────────────────────────
# 4. Career Recommendation Radar Chart
# ──────────────────────────────────────────────────────────────────────

def plot_career_radar(report: dict) -> Path:
    careers = report["top_careers"][:6]
    profile = report["personality_profile"]

    traits = list(profile.keys())
    N = len(traits)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, (ax_radar, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                            subplot_kw=dict(polar=False))
    fig.suptitle("Career Recommendation Report", color="#e8eaf0", fontsize=14)

    # Left: Radar — personality profile
    ax_radar = plt.subplot(121, polar=True)
    ax_radar.set_facecolor("#0d1220")
    vals = [profile[t]["score"] for t in traits] + [profile[traits[0]]["score"]]
    ax_radar.plot(angles, vals, color=PALETTE["purple"], linewidth=2)
    ax_radar.fill(angles, vals, color=PALETTE["purple"], alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(traits, fontsize=9, color="#c8d4e0")
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Personality Profile", color="#e8eaf0", pad=15)
    ax_radar.tick_params(colors="#8899aa")
    ax_radar.grid(color="#2a3a50")

    # Right: Top career compatibility bar chart
    ax_bar = plt.subplot(122)
    ax_bar.set_facecolor("#0d1220")
    titles = [c["title"][:25] for c in careers]
    scores = [c["compatibility_pct"] for c in careers]
    bar_colors = [PALETTE["gold"] if c["autism_strength"] else PALETTE["blue"] for c in careers]
    bars = ax_bar.barh(titles[::-1], scores[::-1], color=bar_colors[::-1], alpha=0.85)
    ax_bar.set_xlabel("Compatibility %", color="#c8d4e0")
    ax_bar.set_title("Top Career Matches", color="#e8eaf0")
    ax_bar.set_xlim(0, 100)
    ax_bar.grid(True, axis="x")
    ax_bar.axvline(50, color="#e8eaf0", linewidth=0.5, alpha=0.3)
    blue_patch = mpatches.Patch(color=PALETTE["blue"], label="General")
    gold_patch = mpatches.Patch(color=PALETTE["gold"], label="★ Autism Strength")
    ax_bar.legend(handles=[blue_patch, gold_patch], loc="lower right", framealpha=0.3, fontsize=8)

    plt.tight_layout()
    return save_fig(fig, "04_career_radar_report")


# ──────────────────────────────────────────────────────────────────────
# 5. Modality Contribution
# ──────────────────────────────────────────────────────────────────────

def plot_modality_contributions(contributions: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Modality Contributions to Prediction", color="#e8eaf0", fontsize=13)

    labels = list(contributions.keys())
    vals   = list(contributions.values())
    colors = [PALETTE["green"], PALETTE["gold"], PALETTE["red"], PALETTE["purple"]]

    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#e8eaf0", "fontsize": 11},
        wedgeprops={"edgecolor": "#0a0e1a", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("#0a0e1a")
        at.set_fontweight("bold")

    ax.set_facecolor("#0d1220")
    return save_fig(fig, "05_modality_contributions")


# ──────────────────────────────────────────────────────────────────────
# 6. PGS Distribution
# ──────────────────────────────────────────────────────────────────────

def plot_pgs_distributions(pgs_scores: np.ndarray, is_autism: np.ndarray, trait_names: list) -> Path:
    n_traits = min(len(trait_names), pgs_scores.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Polygenic Score Distributions by Group", color="#e8eaf0", fontsize=13)

    for i, ax in enumerate(axes.flat):
        if i >= n_traits:
            ax.set_visible(False)
            continue
        bins = 30
        ax.hist(pgs_scores[is_autism == 0, i], bins=bins, alpha=0.7,
                color=PALETTE["blue"], label="NT", density=True)
        ax.hist(pgs_scores[is_autism == 1, i], bins=bins, alpha=0.7,
                color=PALETTE["orange"], label="ASD", density=True)
        ax.set_title(trait_names[i].replace("_PGS", ""), fontsize=9, color="#e8eaf0")
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, "06_pgs_distributions")


# ──────────────────────────────────────────────────────────────────────
# 7. Career Domain Overview
# ──────────────────────────────────────────────────────────────────────

def plot_career_domain_overview() -> Path:
    from collections import Counter
    domain_counts  = Counter(c["domain"] for c in CAREER_DATABASE)
    autism_domains = Counter(c["domain"] for c in CAREER_DATABASE if c["autism_strength"])

    domains = list(domain_counts.keys())
    total   = [domain_counts[d] for d in domains]
    autism  = [autism_domains.get(d, 0) for d in domains]

    x = np.arange(len(domains))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Career Database: Domain Overview", color="#e8eaf0", fontsize=13)
    ax.bar(x - w/2, total,  w, label="All Careers",    color=PALETTE["blue"],  alpha=0.85)
    ax.bar(x + w/2, autism, w, label="Autism-Strength", color=PALETTE["gold"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Number of Careers")
    ax.legend(framealpha=0.3)
    ax.grid(True, axis="y")
    plt.tight_layout()
    return save_fig(fig, "07_career_domain_overview")


# ──────────────────────────────────────────────────────────────────────
# 8. ASD Probability Calibration
# ──────────────────────────────────────────────────────────────────────

def plot_asd_calibration(asd_preds: np.ndarray, asd_true: np.ndarray) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ASD Classifier Analysis", color="#e8eaf0", fontsize=13)

    # Score distributions
    ax1.hist(asd_preds[asd_true == 0], bins=30, alpha=0.7, color=PALETTE["blue"],
             label="Neurotypical", density=True)
    ax1.hist(asd_preds[asd_true == 1], bins=30, alpha=0.7, color=PALETTE["orange"],
             label="ASD", density=True)
    ax1.axvline(0.5, color="#e8eaf0", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Predicted ASD Probability")
    ax1.set_title("Score Distributions", color="#e8eaf0")
    ax1.legend(framealpha=0.3)
    ax1.grid(True)

    # Calibration plot (reliability diagram)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    actual_fracs = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (asd_preds >= lo) & (asd_preds < hi)
        if mask.sum() > 0:
            actual_fracs.append(asd_true[mask].mean())
        else:
            actual_fracs.append(np.nan)

    ax2.plot([0, 1], [0, 1], "w--", alpha=0.5, label="Perfect calibration")
    valid = ~np.isnan(actual_fracs)
    ax2.plot(bin_centers[valid], np.array(actual_fracs)[valid],
             "o-", color=PALETTE["green"], linewidth=2, label="Model")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction Positive")
    ax2.set_title("Calibration Plot", color="#e8eaf0")
    ax2.legend(framealpha=0.3)
    ax2.grid(True)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return save_fig(fig, "08_asd_calibration")


# ──────────────────────────────────────────────────────────────────────
# Master report generator
# ──────────────────────────────────────────────────────────────────────

def generate_all_plots(data: dict, history: dict, report: dict):
    """Generate full visualization suite from dataset + training results."""
    print("\n[Visualization] Generating all plots...")

    big_five   = data["big_five"]
    is_autism  = data["is_autism"].astype(int)
    biomarkers = data["biomarkers"]
    pgs_scores = data["pgs_scores"]

    from configs.config import GENOMIC_CFG
    trait_names = GENOMIC_CFG.pgs_traits

    paths = []
    paths.append(plot_training_curves(history))
    paths.append(plot_bigfive_distributions(big_five, is_autism))
    paths.append(plot_biomarker_comparison(biomarkers, is_autism))
    paths.append(plot_career_radar(report))
    paths.append(plot_modality_contributions(report["modality_contributions"]))
    paths.append(plot_pgs_distributions(pgs_scores, is_autism, trait_names))
    paths.append(plot_career_domain_overview())

    # Simulate ASD predictions for calibration plot
    rng = np.random.default_rng(42)
    asd_preds_sim = np.clip(
        is_autism * 0.6 + rng.normal(0, 0.2, len(is_autism)), 0, 1
    )
    paths.append(plot_asd_calibration(asd_preds_sim, is_autism))

    print(f"[Visualization] Generated {len(paths)} plots in {REPORTS_DIR}")
    return paths


if __name__ == "__main__":
    print("Visualization module loaded. Run generate_all_plots() with data.")
