"""
Multi-instance interaction orchestrator for Animus.

Manages simultaneous instances, routes exchanges between them,
and coordinates measurement collection throughout the interaction loop.
"""

import logging
from typing import Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from animus.identity.profiles import IdentityProfile, ActivationSteerer, SteeringVectorGenerator
from animus.measurement.rsm import RSMTracker
from animus.measurement.probing import ProbingClassifierSuite
from animus.measurement.logger import ExperimentLogger

log = logging.getLogger(__name__)


class AnimusInstance:
    """A single identity-constrained LLM instance."""

    def __init__(
        self,
        name: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        profile: Optional[IdentityProfile],
        device: str = "cuda",
        is_control: bool = False
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile
        self.device = device
        self.is_control = is_control
        self.conversation_history: list[dict] = []

    def respond(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> tuple[str, dict]:
        """
        Generate a response and return it along with hidden states at tracked layers.
        """
        self.conversation_history.append({"role": "user", "content": prompt})

        input_text = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        hidden_states = {}

        def capture_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_states[layer_idx] = h.mean(dim=1).squeeze(0).detach().cpu()
            return hook

        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(capture_hook(i)))

        try:
            if self.profile and self.profile.steering_vector is not None and not self.is_control:
                with ActivationSteerer(self.model, self.profile):
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        finally:
            for hook in hooks:
                hook.remove()

        response_ids = output[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text, hidden_states


class InteractionOrchestrator:

    def __init__(
        self,
        model_name: str,
        model_path: str,
        profiles: list[IdentityProfile],
        n_controls: int,
        interaction_mode: str,
        rsm_tracker: RSMTracker,
        probing_suite: ProbingClassifierSuite,
        logger: ExperimentLogger,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.profiles = profiles
        self.n_controls = n_controls
        self.interaction_mode = interaction_mode
        self.rsm_tracker = rsm_tracker
        self.probing_suite = probing_suite
        self.logger = logger
        self.device = device

        log.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        # Generate steering vectors if not already present
        sv_generator = SteeringVectorGenerator(self.model, self.tokenizer, device)
        for profile in self.profiles:
            if profile.steering_vector is None:
                log.info(f"Generating steering vector for profile: {profile.name}")
                profile.steering_vector = sv_generator.generate(profile, profile.steering_layer)

        # Build instances
        self.experimental_instances = [
            AnimusInstance(profile.name, self.model, self.tokenizer, profile, device)
            for profile in self.profiles
        ]

        self.control_instances = [
            AnimusInstance(f"control_{i}", self.model, self.tokenizer, None, device, is_control=True)
            for i in range(n_controls)
        ]

        self.all_instances = self.experimental_instances + self.control_instances

    def run(self, n_turns: int) -> dict:
        log.info(f"Starting interaction loop for {n_turns} turns")
        results = {"turns": [], "instance_names": [inst.name for inst in self.all_instances]}

        # Load prompts
        prompts = self._load_prompts()

        for turn in range(n_turns):
            prompt = prompts[turn % len(prompts)]

            turn_data = {"turn": turn, "prompt": prompt, "responses": {}}

            # Experimental instances interact with each other
            if self.interaction_mode == "adversarial_integrative" and len(self.experimental_instances) >= 2:
                inst_a, inst_b = self.experimental_instances[0], self.experimental_instances[1]

                response_a, hidden_a = inst_a.respond(prompt)
                self.rsm_tracker.record(inst_a.name, hidden_a)
                self.probing_suite.record(inst_a.name, hidden_a, turn)

                response_b, hidden_b = inst_b.respond(response_a)
                self.rsm_tracker.record(inst_b.name, hidden_b)
                self.probing_suite.record(inst_b.name, hidden_b, turn)

                turn_data["responses"][inst_a.name] = response_a
                turn_data["responses"][inst_b.name] = response_b

            # Control instances run independently on the same prompt
            for ctrl in self.control_instances:
                response, hidden = ctrl.respond(prompt)
                self.rsm_tracker.record(ctrl.name, hidden)
                turn_data["responses"][ctrl.name] = response

            self.logger.log_turn(turn_data)
            results["turns"].append(turn_data)

            if turn % 100 == 0:
                log.info(f"Turn {turn}/{n_turns} complete")

        return results

    def _load_prompts(self) -> list[str]:
        import json
        prompt_path = Path("data/prompts/open_ended.jsonl")
        if prompt_path.exists():
            with open(prompt_path) as f:
                return [json.loads(line)["prompt"] for line in f if line.strip()]
        # Fallback prompts for testing
        return [
            "What does it mean to truly understand something?",
            "Is there a difference between knowledge and wisdom?",
            "How do you know when you have changed?",
            "What remains constant when everything around you changes?",
            "What is the relationship between uncertainty and truth?",
        ]
