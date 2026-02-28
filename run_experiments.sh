#!/bin/bash
# Experiment runner for NanoGPT Slowrun
# Usage:
#   bash run_experiments.sh              # run all phase 1 quick ablations
#   bash run_experiments.sh phase2       # run phase 2 epoch scaling
#   bash run_experiments.sh full         # run the full best config
#   bash run_experiments.sh single <name> <args>  # run a single experiment
set -e

NGPU=$(nvidia-smi -L | wc -l)
RUN="torchrun --standalone --nproc_per_node=$NGPU train.py"

run_one() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo "  Experiment: $name"
    echo "  Args: $@"
    echo "========================================"
    $RUN --wandb-run="$name" "$@"
    echo "  -> $name DONE"
    echo ""
}

# =========================================================================
# PHASE 1: Quick ablations (3 epochs each, ~12 min on 8xH100)
# Goal: find which improvements help, which hurt
# =========================================================================
phase1() {
    echo "=== PHASE 1: Quick ablations (3 epochs each) ==="

    # 1. Original baseline (no improvements)
    run_one "p1-baseline" \
        --num-epochs=3 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 2. All improvements ON (the new defaults)
    run_one "p1-all-improvements" \
        --num-epochs=3

    # 3. Ablation: no label smoothing
    run_one "p1-no-label-smooth" \
        --num-epochs=3 --label-smoothing=0.0

    # 4. Ablation: no EMA
    run_one "p1-no-ema" \
        --num-epochs=3 --ema-decay=0

    # 5. Ablation: no grad clipping
    run_one "p1-no-gradclip" \
        --num-epochs=3 --grad-clip=0

    # 6. Ablation: no warmup
    run_one "p1-no-warmup" \
        --num-epochs=3 --warmup-ratio=0.0

    # 7. With dropout scheduling (0 -> 0.2)
    run_one "p1-dropout-sched" \
        --num-epochs=3 --dropout-schedule --dropout-end=0.2

    # 8. Label smoothing sweep: 0.05
    run_one "p1-ls-0.05" \
        --num-epochs=3 --label-smoothing=0.05

    # 9. Label smoothing sweep: 0.15
    run_one "p1-ls-0.15" \
        --num-epochs=3 --label-smoothing=0.15

    echo "=== PHASE 1 COMPLETE ==="
}

# =========================================================================
# PHASE 2: Epoch scaling with best config (12, 20, 30, 50 epochs)
# Run after reviewing Phase 1 results
# =========================================================================
phase2() {
    echo "=== PHASE 2: Epoch scaling ==="

    run_one "p2-12ep" --num-epochs=12
    run_one "p2-20ep" --num-epochs=20
    run_one "p2-30ep" --num-epochs=30
    run_one "p2-50ep" --num-epochs=50

    echo "=== PHASE 2 COMPLETE ==="
}

# =========================================================================
# FULL: Final run with best config
# =========================================================================
full() {
    echo "=== FULL RUN ==="
    # TODO: fill in best args from phase 1 + phase 2
    run_one "full-best" --num-epochs=30
    echo "=== FULL RUN COMPLETE ==="
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
case "${1:-phase1}" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    full)   full ;;
    single) shift; single "$@" ;;
    *)      echo "Usage: $0 {phase1|phase2|full|single <name> <args...>}" ;;
esac
