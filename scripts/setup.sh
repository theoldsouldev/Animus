#!/bin/bash
set -e

echo "Setting up Animus environment"

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/prompts data/identity_seeds data/probes models runs

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run experiment with: python experiments/identity_divergence/run.py"
