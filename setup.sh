#!/bin/bash
# One-time setup for a new GPU instance
set -e

echo "=== Setting up slowrun ==="

# Install deps
pip install -r requirements.txt

# Copy data if not present
if [ ! -d "fineweb_data" ]; then
    echo "ERROR: fineweb_data/ not found. Copy it from your local machine:"
    echo "  scp -P <port> -r fineweb_data root@<ip>:~/slowrun/"
    exit 1
fi

# Wandb login (interactive)
wandb login

echo "=== Setup complete ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""
echo "Run experiments with: bash run_experiments.sh"
