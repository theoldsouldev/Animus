"""
Animus — Experiment Analysis and Visualisation
Run after experiments/identity_divergence/run.py to generate plots and reports.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_analysis(run_dir: Path) -> dict:
    analysis_path = run_dir / "logs" / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"No analysis found at {analysis_path}")
    with open(analysis_path) as f:
        return json.load(f)


def plot_divergence_vs_drift(analysis: dict, output_dir: Path):
    rsm = analysis["rsm_divergence"]
    drift = analysis["control_drift"]

    traj_exp = rsm.get("trajectory", [])
    traj_ctrl = drift.get("trajectory", [])

    if not traj_exp:
        print("No RSM trajectory data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Animus Identity Divergence Experiment", fontsize=14, fontweight="bold")

    # Plot 1: Divergence vs drift over time
    ax = axes[0]
    turns_exp = [i * 100 for i in range(len(traj_exp))]
    turns_ctrl = [i * 100 for i in range(len(traj_ctrl))]
    ax.plot(turns_exp, traj_exp, label="Experimental (skeptic vs synthesiser)", color="#4A90D9", linewidth=2)
    ax.plot(turns_ctrl, traj_ctrl, label="Control (unconstrained pair)", color="#E0884A", linewidth=2, linestyle="--")
    ax.set_xlabel("Interaction turns")
    ax.set_ylabel("RSM distance (Frobenius)")
    ax.set_title("Representational Divergence Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Probing accuracy by phase
    ax = axes[1]
    probing = analysis.get("probing_accuracy", {})
    phases = probing.get("by_phase", {})
    if phases:
        phase_names = ["early", "mid", "late"]
        phase_vals = [phases.get(p, 0.5) for p in phase_names]
        colors = ["#A8D5A2", "#5BAD6F", "#2D7D46"]
        bars = ax.bar(phase_names, phase_vals, color=colors, width=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance baseline")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Classifier accuracy")
        ax.set_title("Probing Classifier Accuracy by Phase")
        ax.legend()
        for bar, val in zip(bars, phase_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=11)

    plt.tight_layout()
    plot_path = output_dir / "divergence_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()


def print_report(analysis: dict):
    rsm = analysis["rsm_divergence"]
    drift = analysis["control_drift"]
    probing = analysis.get("probing_accuracy", {})
    self_sim = analysis.get("self_similarity", {})

    ratio = rsm["final"] / max(drift["final"], 1e-8)

    print("\n" + "=" * 50)
    print("ANIMUS EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"\nRSM Divergence (experimental final):  {rsm['final']:.4f}")
    print(f"Control Drift (final):                 {drift['final']:.4f}")
    print(f"Divergence / Drift ratio:              {ratio:.2f}x")
    print(f"RSM trend:                             {rsm.get('trend', 'unknown')}")
    print(f"\nProbing accuracy (late phase):         {probing.get('final', 0.5):.3f}")
    print(f"Probing trend:                         {probing.get('trend', 'unknown')}")
    print(f"By phase: {probing.get('by_phase', {})}")
    print(f"\nSelf-similarity asymmetry:             {self_sim.get('asymmetry', 0.0):.4f}")

    print("\n" + "-" * 50)
    thresholds = [
        ("RSM divergence > 1.5x control drift", ratio > 1.5),
        ("Probing accuracy trend increasing", probing.get("trend") == "increasing"),
        ("Self-similarity asymmetry positive", self_sim.get("asymmetry", 0.0) > 0)
    ]

    all_pass = all(t[1] for t in thresholds)
    for label, passed in thresholds:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    print("\n" + "=" * 50)
    if all_pass:
        print("CONCLUSION: Positive result.")
        print("Directional identity divergence detected above control baseline.")
        print("Consistent with identity formation through relational interaction.")
    else:
        print("CONCLUSION: Negative result.")
        print("Insufficient evidence for relational identity formation at this scale.")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyse Animus experiment results")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to experiment run directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_data = load_analysis(run_dir)
    analysis = analysis_data.get("analysis", analysis_data)

    print_report(analysis)

    if args.plot:
        plot_divergence_vs_drift(analysis, run_dir)


if __name__ == "__main__":
    main()
