"""
Animus — Identity Divergence Experiment
Entry point for running the core identity formation experiment.

Hypothesis: Sustained relational interaction between constrained LLM instances
produces structural divergence in activation space that is directional and stable
rather than random parameter drift.
"""

import argparse
import yaml
import logging
from pathlib import Path

from animus.identity.profiles import IdentityProfile, load_profiles
from animus.interaction.orchestrator import InteractionOrchestrator
from animus.measurement.rsm import RSMTracker
from animus.measurement.probing import ProbingClassifierSuite
from animus.measurement.logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Animus identity divergence experiment")
    parser.add_argument("--config", type=str, default="experiments/identity_divergence/config.yaml")
    parser.add_argument("--output", type=str, default="runs/")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and setup without running")
    return parser.parse_args()


def run_experiment(config: dict, output_dir: Path):
    log.info("Initialising Animus identity divergence experiment")
    log.info(f"Model: {config['model']['name']}")
    log.info(f"Instances: {config['experiment']['n_instances']} experimental + {config['experiment']['n_controls']} control")
    log.info(f"Turns: {config['experiment']['n_turns']}")

    # Load identity profiles for experimental instances
    profiles = load_profiles(config["identities"])
    log.info(f"Loaded {len(profiles)} identity profiles: {[p.name for p in profiles]}")

    # Initialise measurement tools
    rsm_tracker = RSMTracker(
        snapshot_interval=config["measurement"]["rsm_snapshot_interval"],
        layers=config["measurement"]["layers_to_track"],
        output_dir=output_dir / "rsm"
    )

    probing_suite = ProbingClassifierSuite(
        probe_dimensions=config["measurement"]["probe_dimensions"],
        output_dir=output_dir / "probing"
    )

    experiment_logger = ExperimentLogger(output_dir=output_dir / "logs")

    # Initialise orchestrator
    orchestrator = InteractionOrchestrator(
        model_name=config["model"]["name"],
        model_path=config["model"]["path"],
        profiles=profiles,
        n_controls=config["experiment"]["n_controls"],
        interaction_mode=config["experiment"]["interaction_mode"],
        rsm_tracker=rsm_tracker,
        probing_suite=probing_suite,
        logger=experiment_logger
    )

    log.info("Starting interaction loop")
    results = orchestrator.run(n_turns=config["experiment"]["n_turns"])

    log.info("Running final analysis")
    analysis = {
        "rsm_divergence": rsm_tracker.compute_divergence_over_time(),
        "control_drift": rsm_tracker.compute_control_drift(),
        "probing_accuracy": probing_suite.evaluate_all(),
        "self_similarity": rsm_tracker.compute_asymmetric_self_similarity()
    }

    experiment_logger.save_analysis(analysis)
    log.info(f"Experiment complete. Results saved to {output_dir}")

    return analysis


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log.info("Dry run complete. Config valid.")
        return

    analysis = run_experiment(config, output_dir)

    # Print summary
    print("\n--- Experiment Summary ---")
    rsm = analysis["rsm_divergence"]
    drift = analysis["control_drift"]
    print(f"Experimental RSM divergence (final): {rsm['final']:.4f}")
    print(f"Control drift (final):               {drift['final']:.4f}")
    print(f"Divergence / drift ratio:            {rsm['final'] / max(drift['final'], 1e-8):.2f}x")
    print(f"Probing accuracy (final turn):       {analysis['probing_accuracy']['final']:.3f}")
    print(f"Self-similarity asymmetry:           {analysis['self_similarity']['asymmetry']:.4f}")
    print("--------------------------\n")

    if rsm["final"] > drift["final"] * 1.5 and analysis["probing_accuracy"]["trend"] == "increasing":
        print("RESULT: Directional identity divergence detected above control baseline.")
        print("        Consistent with identity formation through relational interaction.")
    else:
        print("RESULT: No significant divergence above control baseline detected.")
        print("        Insufficient evidence for relational identity formation at this scale.")


if __name__ == "__main__":
    main()
