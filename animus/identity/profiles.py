"""
Identity profiles and activation steering for Animus instances.

Each identity profile defines a constrained processing style enforced through
activation steering at a specified layer. The goal is not a surface persona
but a structural bias in how the instance processes and responds to inputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import torch
import numpy as np


@dataclass
class IdentityProfile:
    name: str
    description: str
    steering_layer: int
    steering_coefficient: float
    seed_exchanges_path: Optional[str] = None
    steering_vector_path: Optional[str] = None
    steering_vector: Optional[torch.Tensor] = field(default=None, repr=False)

    def load_steering_vector(self, device: str = "cuda"):
        if self.steering_vector_path and Path(self.steering_vector_path).exists():
            self.steering_vector = torch.load(self.steering_vector_path, map_location=device)
        return self

    def save_steering_vector(self, path: str):
        if self.steering_vector is not None:
            torch.save(self.steering_vector, path)
            self.steering_vector_path = path

    def load_seed_exchanges(self) -> list[dict]:
        if not self.seed_exchanges_path:
            return []
        path = Path(self.seed_exchanges_path)
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]


def load_profiles(profile_configs: list[dict]) -> list[IdentityProfile]:
    profiles = []
    for cfg in profile_configs:
        profile = IdentityProfile(
            name=cfg["name"],
            description=cfg["description"],
            steering_layer=cfg["steering_layer"],
            steering_coefficient=cfg["steering_coefficient"],
            seed_exchanges_path=cfg.get("seed_exchanges"),
            steering_vector_path=cfg.get("steering_vector_path")
        )
        profile.load_steering_vector()
        profiles.append(profile)
    return profiles


class SteeringVectorGenerator:
    """
    Generates activation steering vectors for identity profiles by contrasting
    activations on identity-positive vs identity-negative seed exchanges.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, profile: IdentityProfile, layer: int) -> torch.Tensor:
        seed_exchanges = profile.load_seed_exchanges()
        if not seed_exchanges:
            raise ValueError(f"No seed exchanges found for profile '{profile.name}'")

        positive_activations = []
        negative_activations = []

        for exchange in seed_exchanges:
            pos_acts = self._get_layer_activations(exchange["positive"], layer)
            neg_acts = self._get_layer_activations(exchange["negative"], layer)
            positive_activations.append(pos_acts)
            negative_activations.append(neg_acts)

        pos_mean = torch.stack(positive_activations).mean(dim=0)
        neg_mean = torch.stack(negative_activations).mean(dim=0)

        steering_vector = pos_mean - neg_mean
        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

        return steering_vector

    def _get_layer_activations(self, text: str, layer: int) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        return hidden_states.mean(dim=1).squeeze(0)


class ActivationSteerer:
    """
    Hooks into model forward pass to apply identity steering vectors
    at the specified layer during inference.
    """

    def __init__(self, model, profile: IdentityProfile):
        self.model = model
        self.profile = profile
        self.hook = None

    def __enter__(self):
        self._register_hook()
        return self

    def __exit__(self, *args):
        self._remove_hook()

    def _register_hook(self):
        layer = self.model.model.layers[self.profile.steering_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if self.profile.steering_vector is not None:
                vec = self.profile.steering_vector.to(hidden.device)
                hidden = hidden + self.profile.steering_coefficient * vec
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.hook = layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
