#!/bin/bash
# Experiment runner for NanoGPT Slowrun
# Usage:
#   bash run_experiments.sh              # run leaderboard attempt (default)
#   bash run_experiments.sh leaderboard  # reproduce baseline + try to beat it
#   bash run_experiments.sh ablation     # quick 1-epoch ablations
#   bash run_experiments.sh single <name> <args>  # run a single experiment
set -e

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -gt 1 ]; then
    RUN="torchrun --standalone --nproc_per_node=$NGPU train.py"
else
    RUN="python train.py"
fi

run_one() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo "  Experiment: $name"
    echo "  GPUs: $NGPU | Runner: $RUN"
    echo "  Args: $@"
    echo "========================================"
    $RUN --wandb-run="$name" "$@"
    echo "  -> $name DONE"
    echo ""
}

# =========================================================================
# LEADERBOARD: Get on the board with reliable improvements
# Strategy: baseline already has shuffling. Train longer + try safe tweaks.
# On 1xH100: ~32 min/epoch, 12 epochs = ~6.4h, 20 epochs = ~10.6h
# =========================================================================
leaderboard() {
    echo "=== LEADERBOARD RUNS ==="

    # Run 1: Pure baseline at 12 epochs (should match leaderboard 3.376)
    # No improvements, just the baseline with shuffling (already in DataLoader)
    run_one "lb-baseline-12ep" \
        --num-epochs=12 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Run 2: Baseline at 20 epochs (unlimited track allows more compute)
    run_one "lb-baseline-20ep" \
        --num-epochs=20 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Run 3: 20 epochs + label smoothing 0.05 (conservative, eval is now raw CE)
    run_one "lb-ls005-20ep" \
        --num-epochs=20 --label-smoothing=0.05 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    echo "=== LEADERBOARD RUNS COMPLETE ==="
}

# =========================================================================
# ABLATION: Quick 1-epoch tests to measure individual improvements
# =========================================================================
ablation() {
    echo "=== ABLATIONS (1 epoch each) ==="

    # Baseline (no improvements)
    run_one "abl-baseline" \
        --num-epochs=1 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Label smoothing 0.05
    run_one "abl-ls005" \
        --num-epochs=1 --label-smoothing=0.05 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Label smoothing 0.1
    run_one "abl-ls010" \
        --num-epochs=1 --label-smoothing=0.1 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Warmup 2%
    run_one "abl-warmup" \
        --num-epochs=1 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.02 --swa-start-frac=0 --dropout=0.1

    # Grad clip 1.0
    run_one "abl-gradclip" \
        --num-epochs=1 --label-smoothing=0.0 --ema-decay=0 --grad-clip=1.0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # Dropout 0.15 (higher than baseline 0.1)
    run_one "abl-drop015" \
        --num-epochs=1 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.15

    echo "=== ABLATIONS COMPLETE ==="
}

# =========================================================================
# Single experiment
# =========================================================================
single() {
    local name="$1"
    shift
    run_one "$name" "$@"
}

# =========================================================================
# Main
# =========================================================================
case "${1:-leaderboard}" in
    leaderboard) leaderboard ;;
    ablation)    ablation ;;
    single)      shift; single "$@" ;;
    *)           echo "Usage: $0 {leaderboard|ablation|single <name> <args...>}" ;;
esac
