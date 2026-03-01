#!/bin/bash
# Experiment runner for NanoGPT Slowrun — Limited Track
# Target: beat 3.376 val loss on 8xH100 in <1 hour
# Develop on 1xH100, submit on 8xH100
#
# Usage:
#   bash run_experiments.sh              # run ablations (default)
#   bash run_experiments.sh submit       # final submission run (8xH100 only)
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

# Baseline args: matches the leaderboard #2 config exactly (no improvements)
BASELINE="--label-smoothing=0.0 --ema-decay=0 --grad-clip=0 --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1"

# =========================================================================
# ABLATIONS: 3-epoch runs on 1xH100 (~1.5h each)
# Each changes exactly ONE thing vs baseline.
# Compare val loss at epoch 3 to see what helps.
# =========================================================================
ablation() {
    echo "=== ABLATIONS (3 epochs each, 1 thing changed per run) ==="

    # 0. Baseline reference (shuffling already built in)
    run_one "abl-baseline-3ep" \
        --num-epochs=3 $BASELINE

    # 1. More epochs: 15 instead of 12 (fits in 1hr on 8xH100)
    #    Skip for ablation, test in submit phase

    # 2. Label smoothing 0.05 (mild, eval is raw CE now)
    run_one "abl-ls005-3ep" \
        --num-epochs=3 --label-smoothing=0.05 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 3. Higher dropout 0.15 (more regularization)
    run_one "abl-drop015-3ep" \
        --num-epochs=3 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.15

    # 4. Higher dropout 0.2
    run_one "abl-drop020-3ep" \
        --num-epochs=3 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.2

    # 5. Higher weight decay 2.0 (baseline is 1.6, paper says up to 30x helps)
    run_one "abl-wd20-3ep" \
        --num-epochs=3 --weight-decay=2.0 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 6. Higher weight decay 2.5
    run_one "abl-wd25-3ep" \
        --num-epochs=3 --weight-decay=2.5 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    echo "=== ABLATIONS COMPLETE ==="
    echo "Compare val/bpb at epoch 3 on wandb to find best config"
}

# =========================================================================
# SUBMIT: Final run on 8xH100 — must finish in <1 hour
# Update args below based on ablation results
# =========================================================================
submit() {
    if [ "$NGPU" -lt 8 ]; then
        echo "WARNING: submit mode is designed for 8xH100. You have $NGPU GPU(s)."
        echo "Results won't match leaderboard timing. Continue anyway? [y/N]"
        read -r ans
        [ "$ans" != "y" ] && exit 1
    fi
    echo "=== SUBMISSION RUN ==="
    # TODO: update with best config from ablations
    # Default: baseline + 15 epochs (fits in ~1hr on 8xH100)
    run_one "submit-v1" \
        --num-epochs=15 $BASELINE
    echo "=== SUBMISSION COMPLETE ==="
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
case "${1:-ablation}" in
    ablation) ablation ;;
    submit)   submit ;;
    single)   shift; single "$@" ;;
    *)        echo "Usage: $0 {ablation|submit|single <name> <args...>}" ;;
esac
