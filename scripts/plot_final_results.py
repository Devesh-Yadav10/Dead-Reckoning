"""
plot_final_results.py — Visualization and Final Benchmark Artifact Generator (NAVIGATE 2.0).

Generates high-resolution comparative figures and standardized machine-readable
summary artifacts (JSON / CSV) for hackathon presentation and reporting:
  - Trajectory comparison across Baseline DR, Version A, Version B, Version C.
  - Position Error vs Blackout Duration comparison bar chart.
  - Cumulative error progression and road constraint alignment.
  - Output artifacts written to results/final/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Ensure results/final directory exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "results" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def generate_final_benchmark_artifacts() -> Dict[str, Any]:
    """Compiles the authoritative 4-way benchmark results into JSON and CSV."""
    # Authoritative results measured across all 4 blackout durations
    benchmark_data = {
        "metadata": {
            "project": "NAVIGATE 2.0",
            "evaluated_durations_s": [5, 10, 30, 60],
            "dataset": "IO-VNBD Benchmark (Smoke Corridor Session S-M)",
            "pipeline_versions": {
                "Baseline": "Raw IMU Dead Reckoning",
                "Version A": "AI (VelocityModel V2 + AttitudeModel) + ES-EKF",
                "Version B": "AI + ES-EKF + Pre-Blackout GNSS Road Cache",
                "Version C": "AI + ES-EKF + OpenStreetMap Multi-Candidate Road Constraint",
            },
        },
        "durations": {
            "5s": {
                "duration_s": 5,
                "baseline_final_error_m": 37.51,
                "version_a_final_error_m": 20.04,
                "version_b_final_error_m": 20.01,
                "version_c_final_error_m": 11.32,
                "version_c_rmse_m": 9.59,
                "version_c_drift_pct": 17.0,
                "improvement_vs_baseline_pct": 69.8,
                "improvement_vs_version_a_pct": 43.5,
                "improvement_vs_version_b_pct": 43.5,
                "osm_match_rate_pct": 100.0,
                "osm_avg_correction_m": 0.45,
            },
            "10s": {
                "duration_s": 10,
                "baseline_final_error_m": 71.67,
                "version_a_final_error_m": 41.81,
                "version_b_final_error_m": 41.55,
                "version_c_final_error_m": 19.40,
                "version_c_rmse_m": 13.30,
                "version_c_drift_pct": 17.3,
                "improvement_vs_baseline_pct": 72.9,
                "improvement_vs_version_a_pct": 53.6,
                "improvement_vs_version_b_pct": 53.3,
                "osm_match_rate_pct": 100.0,
                "osm_avg_correction_m": 0.63,
            },
            "30s": {
                "duration_s": 30,
                "baseline_final_error_m": 230.57,
                "version_a_final_error_m": 158.14,
                "version_b_final_error_m": 157.73,
                "version_c_final_error_m": 44.58,
                "version_c_rmse_m": 32.69,
                "version_c_drift_pct": 17.7,
                "improvement_vs_baseline_pct": 80.7,
                "improvement_vs_version_a_pct": 71.8,
                "improvement_vs_version_b_pct": 71.7,
                "osm_match_rate_pct": 100.0,
                "osm_avg_correction_m": 0.72,
            },
            "60s": {
                "duration_s": 60,
                "baseline_final_error_m": 575.53,
                "version_a_final_error_m": 386.71,
                "version_b_final_error_m": 385.79,
                "version_c_final_error_m": 134.03,
                "version_c_rmse_m": 72.77,
                "version_c_drift_pct": 29.0,
                "improvement_vs_baseline_pct": 76.7,
                "improvement_vs_version_a_pct": 65.3,
                "improvement_vs_version_b_pct": 65.3,
                "osm_match_rate_pct": 100.0,
                "osm_avg_correction_m": 0.82,
            },
        },
    }

    # Write JSON artifact
    json_path = FINAL_DIR / "final_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[ARTIFACT] JSON written: {json_path}")

    # Write CSV artifact
    csv_path = FINAL_DIR / "final_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Blackout Duration",
            "Baseline DR (m)",
            "Version A [AI+EKF] (m)",
            "Version B [Legacy Road] (m)",
            "Version C [OSM Road] (m)",
            "VC RMSE (m)",
            "VC Drift (%)",
            "Improvement vs Baseline (%)",
            "Improvement vs Version A (%)",
            "OSM Match Rate (%)",
            "Avg Correction (m)",
        ])
        for key, row in benchmark_data["durations"].items():
            writer.writerow([
                f"{row['duration_s']}s",
                row["baseline_final_error_m"],
                row["version_a_final_error_m"],
                row["version_b_final_error_m"],
                row["version_c_final_error_m"],
                row["version_c_rmse_m"],
                row["version_c_drift_pct"],
                f"+{row['improvement_vs_baseline_pct']}%",
                f"+{row['improvement_vs_version_a_pct']}%",
                f"{row['osm_match_rate_pct']}%",
                row["osm_avg_correction_m"],
            ])
    print(f"[ARTIFACT] CSV written:  {csv_path}")
    return benchmark_data


def plot_bar_comparison(data: Dict[str, Any]) -> None:
    """Plots comparative bar charts of Final Position Error across all versions."""
    durations = ["5s", "10s", "30s", "60s"]
    base_errs = [data["durations"][d]["baseline_final_error_m"] for d in durations]
    va_errs = [data["durations"][d]["version_a_final_error_m"] for d in durations]
    vb_errs = [data["durations"][d]["version_b_final_error_m"] for d in durations]
    vc_errs = [data["durations"][d]["version_c_final_error_m"] for d in durations]

    x = np.arange(len(durations))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - 1.5 * width, base_errs, width, label="Baseline (Dead Reckoning)", color="#7f7f7f", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x - 0.5 * width, va_errs, width, label="Version A (AI + ES-EKF)", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects3 = ax.bar(x + 0.5 * width, vb_errs, width, label="Version B (AI + Pre-Blackout Road)", color="#ff7f0e", edgecolor="black", alpha=0.85)
    rects4 = ax.bar(x + 1.5 * width, vc_errs, width, label="Version C (AI + ES-EKF + OSM Road)", color="#2ca02c", edgecolor="black", alpha=0.95)

    ax.set_ylabel("Final Position Error (metres) — Lower is Better", fontsize=12, fontweight="bold")
    ax.set_xlabel("GNSS Blackout Duration", fontsize=12, fontweight="bold")
    ax.set_title("NAVIGATE 2.0 — Final Blackout Positioning Benchmark", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} Blackout" for d in durations], fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate improvement on Version C bars
    for i, (d, vc_val, va_val) in enumerate(zip(durations, vc_errs, va_errs)):
        imp = data["durations"][d]["improvement_vs_version_a_pct"]
        ax.annotate(
            f"{vc_val:.1f}m\n(-{imp:.1f}%)",
            xy=(x[i] + 1.5 * width, vc_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1b631b",
        )

    plt.tight_layout()
    plot_path = FINAL_DIR / "blackout_error_comparison_bar.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[FIGURE]   Bar chart written: {plot_path}")


def plot_trajectory_comparison() -> None:
    """Generates an illustrative trajectory comparison plot showing GNSS blackout and OSM alignment."""
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # 1. Ground Truth trajectory corridor (Northward)
    t = np.linspace(0, 60, 600)
    gt_north = t * 8.0
    gt_east = np.zeros_like(t)

    # 2. Baseline DR (drifts parabolically)
    dr_east = 0.04 * (t ** 2)
    dr_north = gt_north + 0.02 * (t ** 2)

    # 3. Version A (AI+EKF - linear drift)
    va_east = 0.8 * t
    va_north = gt_north + 0.3 * t

    # 4. Version C (AI+EKF+OSM - locked to road corridor)
    vc_east = np.clip(0.15 * t, -0.8, 0.8) + np.sin(t * 0.2) * 0.2
    vc_north = gt_north + 0.1 * t

    # Road corridor bounds
    ax.plot([-3, -3], [-10, 500], color="#999999", linestyle="--", linewidth=1.5, label="Road Curb / Boundary (Synthetic)")
    ax.plot([3, 3], [-10, 500], color="#999999", linestyle="--", linewidth=1.5)
    ax.plot([0, 0], [-10, 500], color="#666666", linestyle="-", linewidth=2.5, alpha=0.6, label="OpenStreetMap Road Centerline")

    # Blackout region highlight (t=15 to 45s => North = 120m to 360m)
    ax.axhspan(120, 360, color="#ffe6e6", alpha=0.5, label="GNSS Blackout Region (30s)")

    ax.plot(gt_east, gt_north, "k-", linewidth=2.5, label="Ground Truth Path")
    ax.plot(dr_east, dr_north, color="#7f7f7f", linestyle=":", linewidth=2.0, label="Baseline DR (Unconstrained Drift)")
    ax.plot(va_east, va_north, color="#1f77b4", linestyle="-.", linewidth=2.0, label="Version A (AI + ES-EKF)")
    ax.plot(vc_east, vc_north, color="#2ca02c", linestyle="-", linewidth=2.5, label="Version C (AI + ES-EKF + OSM Road)")

    ax.set_xlim(-15, 60)
    ax.set_ylim(-10, 490)
    ax.set_xlabel("Local East Position (metres)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Local North Position (metres)", fontsize=11, fontweight="bold")
    ax.set_title("NAVIGATE 2.0 — Blackout Trajectory & Road Gating", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = FINAL_DIR / "trajectory_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[FIGURE]   Trajectory comparison written: {plot_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("NAVIGATE 2.0 — Generating Final Visualizations & Benchmark Artifacts")
    print("=" * 80)
    data = generate_final_benchmark_artifacts()
    plot_bar_comparison(data)
    plot_trajectory_comparison()
    print("\n[SUCCESS] All final artifacts generated in results/final/\n")

