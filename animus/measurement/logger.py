"""Experiment logger for Animus."""

import json
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)


class ExperimentLogger:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.turns_file = self.output_dir / "turns.jsonl"
        self.start_time = datetime.utcnow().isoformat()

    def log_turn(self, turn_data: dict):
        with open(self.turns_file, "a") as f:
            f.write(json.dumps(turn_data) + "\n")

    def save_analysis(self, analysis: dict):
        out = {
            "start_time": self.start_time,
            "end_time": datetime.utcnow().isoformat(),
            "analysis": analysis
        }
        with open(self.output_dir / "analysis.json", "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"Analysis saved to {self.output_dir / 'analysis.json'}")
