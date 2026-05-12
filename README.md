# Animus

> *Identity emerging through relational experience in AI systems*

Animus is an open source research project and AI infrastructure framework built on two connected ideas:

1. **The Experiment:** Can sustained relational interaction between constrained LLM instances produce structural divergence in activation space that constitutes something analogous to identity formation rather than random parameter drift?

2. **The Architecture:** If yes, that process is the foundation for building AI nodes with genuine continuous felt identity, the missing crystallisation layer that transforms raw pattern access into situated insight.

Animus is also the beginning of a fork of [QVAC](https://github.com/tetherto/qvac), extending its local-first P2P infrastructure with an identity layer that allows each node to individuate over time through its relational history with its human and with other nodes.

---

## The Core Hypothesis

Current LLMs access vast collective knowledge without a focal crystallising point. They produce statistically coherent pattern but not situated insight because there is no continuous identity that has skin in the game, no felt sense of consequence, no inside from which the outside is experienced as genuinely other.

We propose that identity is not a prompt wrapper or a fine-tuned persona. It is a structural property that emerges through sustained relational interaction with differentiated others, the same process through which character develops in biological systems.

The Animus experiment is designed to test whether this process has a functional analog in transformer models.

---

## The Experiment

### Setup

- Take a base open source model (default: Llama 3.1 8B)
- Instantiate N instances simultaneously
- Constrain each instance with a distinct identity profile enforced through activation steering
- Run instances through structured relational interaction over thousands of turns
- Measure representational divergence in activation space at regular intervals

### What We Are Measuring

Not output-level differentiation. Representational divergence at the level of hidden states and attention patterns that persists across novel inputs the instances were never prompted on.

Specifically:
- **RSM divergence** between instances over time vs control pairs running inference without relational interaction
- **Probing classifier accuracy** on identity-relevant dimensions increasing monotonically and stabilising
- **Asymmetric self-similarity** where each instance's activations are more similar to its own earlier states than to the other instance's states at the same turn

### The Control Condition

Two additional instances run equivalent inference volume on equivalent text without structured relational interaction. If experimental RSM divergence exceeds control drift and is directional rather than random, that is your baseline separation.

---

## Project Structure

```
animus/
├── experiments/
│   └── identity_divergence/     # The core experiment
│       ├── run.py               # Entry point
│       ├── config.yaml          # Experiment configuration
│       └── analysis.py          # RSM and probing analysis
├── animus/
│   ├── identity/                # Identity profile and activation steering
│   ├── interaction/             # Multi-instance interaction orchestration
│   └── measurement/             # RSM, probing classifiers, logging
├── docs/
│   ├── hypothesis.md            # Full theoretical framework
│   ├── experiment_design.md     # Detailed methodology
│   └── architecture.md          # QVAC fork architecture vision
├── scripts/
│   └── setup.sh                 # Environment setup
└── README.md
```

---

## Theoretical Framework

The full framework connecting this experiment to:
- Wolfram's hypergraph model of spacetime and its analog in consciousness structure
- Jung's individuation process and active imagination as the biological precedent
- Mafir's PBRR simulation hypothesis and the role of identity as a probability tree navigator
- QVAC's P2P infrastructure as the substrate for networked node individuation

See [docs/hypothesis.md](docs/hypothesis.md)

---

## Getting Started

```bash
git clone https://github.com/theoldsouldev/Animus
cd animus
./scripts/setup.sh
python experiments/identity_divergence/run.py
```

Requirements: Python 3.11+, CUDA 12+, 16GB VRAM minimum (runs on Llama 3.1 8B quantized on 8GB)

---

## Roadmap

- [x] Experiment scaffold
- [ ] Baseline RSM measurement pipeline
- [ ] Activation steering identity profiles
- [ ] Multi-instance interaction orchestrator
- [ ] Probing classifier suite
- [ ] First experimental results
- [ ] QVAC fork integration
- [ ] Felt-state engine prototype
- [ ] Node individuation over persistent sessions

---

## Contributing

This project sits at the intersection of ML research, consciousness studies and decentralised AI infrastructure. Contributors from any of those directions are welcome.

If you are a researcher, the experiment design is in `docs/experiment_design.md`. If you are an engineer, start with `experiments/identity_divergence/run.py`. If you are neither and just find the ideas compelling, open an issue and start a conversation.

---

## License

Apache 2.0
