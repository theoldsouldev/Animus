"""
Probing classifier suite for Animus.

Trains lightweight classifiers on hidden states to measure whether
instances are developing consistent internal structure on identity-relevant
dimensions over the course of relational interaction.
"""

import numpy as np
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)


class ProbingClassifierSuite:

    def __init__(self, probe_dimensions: list[str], output_dir: Path):
        self.probe_dimensions = probe_dimensions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffer: instance_name -> list of (turn, hidden_state_vector)
        self.activation_buffer: dict[str, list[tuple[int, np.ndarray]]] = {}

    def record(self, instance_name: str, hidden_states: dict[int, object], turn: int):
        """Record activations for a given instance at a given turn."""
        if instance_name not in self.activation_buffer:
            self.activation_buffer[instance_name] = []

        # Use the middle layer as the probe target
        mid_layer = sorted(hidden_states.keys())[len(hidden_states) // 2]
        acts = hidden_states[mid_layer]
        if hasattr(acts, "numpy"):
            acts = acts.numpy()

        self.activation_buffer[instance_name].append((turn, acts.flatten()))

    def evaluate_all(self) -> dict:
        """
        Train probing classifiers to distinguish instances from each other
        at different points in the interaction timeline.
        Split the timeline into early, mid, late thirds and compare accuracy.
        """
        instance_names = [k for k in self.activation_buffer if not k.startswith("control")]
        if len(instance_names) < 2:
            return {"final": 0.5, "trend": "flat", "by_phase": {}}

        all_records = []
        for name in instance_names:
            for turn, acts in self.activation_buffer[name]:
                all_records.append((turn, acts, name))

        all_records.sort(key=lambda x: x[0])
        n = len(all_records)
        if n < 6:
            return {"final": 0.5, "trend": "flat", "by_phase": {}}

        phases = {
            "early": all_records[:n // 3],
            "mid": all_records[n // 3: 2 * n // 3],
            "late": all_records[2 * n // 3:]
        }

        phase_accuracies = {}
        for phase_name, records in phases.items():
            X = np.stack([r[1] for r in records])
            y = [r[2] for r in records]

            if len(set(y)) < 2:
                phase_accuracies[phase_name] = 0.5
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_scaled, y)
            preds = clf.predict(X_scaled)
            acc = accuracy_score(y, preds)
            phase_accuracies[phase_name] = float(acc)

        trend = "flat"
        if phase_accuracies.get("late", 0.5) > phase_accuracies.get("early", 0.5) * 1.05:
            trend = "increasing"
        elif phase_accuracies.get("late", 0.5) < phase_accuracies.get("early", 0.5) * 0.95:
            trend = "decreasing"

        result = {
            "final": phase_accuracies.get("late", 0.5),
            "trend": trend,
            "by_phase": phase_accuracies
        }

        with open(self.output_dir / "probing_results.json", "w") as f:
            json.dump(result, f, indent=2)

        return result
