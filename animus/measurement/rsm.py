"""
Representational Similarity Matrix (RSM) tracking for the Animus experiment.

Tracks how the internal representations of each instance evolve over time
and measures divergence between experimental instances vs control drift.
"""

from pathlib import Path
from typing import Optional
import torch
import numpy as np
import json
import logging

log = logging.getLogger(__name__)


class RSMTracker:

    def __init__(self, snapshot_interval: int, layers: list[int], output_dir: Path):
        self.snapshot_interval = snapshot_interval
        self.layers = layers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshots indexed by turn, then instance name, then layer
        self.snapshots: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
        self.turn_counter = 0

    def record(self, instance_name: str, hidden_states: dict[int, torch.Tensor]):
        """Record hidden states for a given instance at the current turn."""
        self.turn_counter += 1

        if self.turn_counter % self.snapshot_interval != 0:
            return

        if self.turn_counter not in self.snapshots:
            self.snapshots[self.turn_counter] = {}

        self.snapshots[self.turn_counter][instance_name] = {
            layer: hidden_states[layer].detach().cpu()
            for layer in self.layers
            if layer in hidden_states
        }

        log.debug(f"RSM snapshot recorded: turn={self.turn_counter} instance={instance_name}")

    def compute_divergence_over_time(self) -> dict:
        """
        Compute pairwise RSM distance between experimental instances at each snapshot.
        Returns final value, trajectory, and whether trend is increasing.
        """
        turns = sorted(self.snapshots.keys())
        divergences = []

        for turn in turns:
            snapshot = self.snapshots[turn]
            instance_names = [k for k in snapshot if not k.startswith("control")]
            if len(instance_names) < 2:
                continue

            layer_divergences = []
            for layer in self.layers:
                acts = [snapshot[name][layer].numpy() for name in instance_names if layer in snapshot[name]]
                if len(acts) < 2:
                    continue
                rsm_dist = self._rsm_distance(acts[0], acts[1])
                layer_divergences.append(rsm_dist)

            if layer_divergences:
                divergences.append(np.mean(layer_divergences))

        if not divergences:
            return {"final": 0.0, "trajectory": [], "trend": "flat"}

        trend = "increasing" if divergences[-1] > divergences[0] * 1.1 else "flat"

        return {
            "final": float(divergences[-1]),
            "trajectory": [float(d) for d in divergences],
            "trend": trend
        }

    def compute_control_drift(self) -> dict:
        """Same computation on control instances."""
        turns = sorted(self.snapshots.keys())
        drifts = []

        for turn in turns:
            snapshot = self.snapshots[turn]
            control_names = [k for k in snapshot if k.startswith("control")]
            if len(control_names) < 2:
                continue

            layer_drifts = []
            for layer in self.layers:
                acts = [snapshot[name][layer].numpy() for name in control_names if layer in snapshot[name]]
                if len(acts) < 2:
                    continue
                rsm_dist = self._rsm_distance(acts[0], acts[1])
                layer_drifts.append(rsm_dist)

            if layer_drifts:
                drifts.append(np.mean(layer_drifts))

        if not drifts:
            return {"final": 0.0, "trajectory": []}

        return {
            "final": float(drifts[-1]),
            "trajectory": [float(d) for d in drifts]
        }

    def compute_asymmetric_self_similarity(self) -> dict:
        """
        For each instance, compare its current activations to its own earlier activations
        vs the other instance's earlier activations at the same turn.
        Higher asymmetry means the instance is more similar to its own past than to the other's past.
        """
        turns = sorted(self.snapshots.keys())
        if len(turns) < 2:
            return {"asymmetry": 0.0}

        first_turn = turns[0]
        last_turn = turns[-1]

        asymmetries = []
        snapshot_first = self.snapshots[first_turn]
        snapshot_last = self.snapshots[last_turn]

        instance_names = [k for k in snapshot_last if not k.startswith("control")]
        if len(instance_names) < 2:
            return {"asymmetry": 0.0}

        for layer in self.layers:
            for i, name in enumerate(instance_names):
                other_name = instance_names[1 - i]
                if name not in snapshot_first or other_name not in snapshot_first:
                    continue
                if layer not in snapshot_last.get(name, {}) or layer not in snapshot_first.get(name, {}):
                    continue

                self_sim = self._cosine_similarity(
                    snapshot_last[name][layer].numpy(),
                    snapshot_first[name][layer].numpy()
                )
                cross_sim = self._cosine_similarity(
                    snapshot_last[name][layer].numpy(),
                    snapshot_first[other_name][layer].numpy()
                )
                asymmetries.append(float(self_sim - cross_sim))

        return {"asymmetry": float(np.mean(asymmetries)) if asymmetries else 0.0}

    def _rsm_distance(self, acts_a: np.ndarray, acts_b: np.ndarray) -> float:
        """Frobenius distance between RSMs of two activation matrices."""
        rsm_a = np.corrcoef(acts_a) if acts_a.ndim > 1 else np.array([[1.0]])
        rsm_b = np.corrcoef(acts_b) if acts_b.ndim > 1 else np.array([[1.0]])
        return float(np.linalg.norm(rsm_a - rsm_b, "fro"))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a_flat, b_flat) / denom)

    def save(self):
        out = {
            str(turn): {
                name: {str(layer): acts.tolist() for layer, acts in layers.items()}
                for name, layers in instances.items()
            }
            for turn, instances in self.snapshots.items()
        }
        with open(self.output_dir / "rsm_snapshots.json", "w") as f:
            json.dump(out, f)
