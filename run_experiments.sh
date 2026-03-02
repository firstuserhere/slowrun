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
# Always use torchrun — the optimizer requires distributed init even for 1 GPU
RUN="torchrun --standalone --nproc_per_node=$NGPU train.py"

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

# Baseline args (no extra tricks — architecture changes are in train.py now)
# SwiGLU + VE projections + WD 1.4 are all defaults in train.py
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
# ABLATION PHASE 2: WD extension + training tricks
# Run after Phase 1 results are in. Tests WD 3.0-5.0 and other Tier 1 ideas.
# Each still changes ONE thing vs baseline (except combo runs at the end).
# =========================================================================
ablation2() {
    echo "=== ABLATION PHASE 2 (3 epochs each) ==="
    echo "Phase 1 results: WD 2.0 hurt, dropout hurt, label smoothing hurt."
    echo "Phase 2 focuses on: batch size, warmup, cyclic SWA, EMA, fine WD."

    # 1. Smaller batch size (262144 — 2x more gradient steps per epoch)
    run_one "abl-batch262k-3ep" \
        --num-epochs=3 --total-batch-size=262144 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 2. Even smaller batch (131072 — 4x more gradient steps)
    run_one "abl-batch131k-3ep" \
        --num-epochs=3 --total-batch-size=131072 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 3. Cyclic SWA warmdown (6 cosine cycles, collects ~6 checkpoints and averages)
    run_one "abl-cyclic6-3ep" \
        --num-epochs=3 --cyclic-warmdown=6 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --dropout=0.1

    # 4. 2% warmup (baseline uses 0%)
    run_one "abl-warmup002-3ep" \
        --num-epochs=3 --warmup-ratio=0.02 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --swa-start-frac=0 --dropout=0.1

    # 5. EMA enabled (decay 0.99)
    run_one "abl-ema099-3ep" \
        --num-epochs=3 --ema-decay=0.99 --label-smoothing=0.0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    # 6. Fine WD sweep: 1.4 (check if lower WD is better)
    run_one "abl-wd14-3ep" \
        --num-epochs=3 --weight-decay=1.4 --label-smoothing=0.0 --ema-decay=0 --grad-clip=0 \
        --warmup-ratio=0.0 --swa-start-frac=0 --dropout=0.1

    echo "=== ABLATION PHASE 2 COMPLETE ==="
    echo "Compare val loss at epoch 3 on wandb."
    echo "Next: combine winners for submission run."
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
    case "${1:-swiglu}" in
        swiglu)
            echo "Config: SwiGLU + WD 1.2 (12 epochs)"
            run_one "submit-swiglu-wd12" \
                --num-epochs=12 --swiglu --weight-decay=1.2 $BASELINE
            ;;
        swiglu14)
            echo "Config: SwiGLU + WD 1.4 (12 epochs)"
            run_one "submit-swiglu-wd14-v2" \
                --num-epochs=12 --swiglu $BASELINE
            ;;
        veproj)
            echo "Config: VE proj + WD 1.4 (15 epochs)"
            run_one "submit-veproj-wd14" \
                --num-epochs=15 --ve-proj $BASELINE
            ;;
        baseline)
            echo "Config: Baseline + WD 1.4 (15 epochs)"
            run_one "submit-baseline-wd14" \
                --num-epochs=15 $BASELINE
            ;;
        combo)
            echo "Config: SwiGLU + VE proj + WD 1.2 (12 epochs)"
            run_one "submit-combo-swiglu-veproj-wd12" \
                --num-epochs=12 --swiglu --ve-proj --weight-decay=1.2 $BASELINE
            ;;
        *)
            echo "Usage: $0 submit {combo|swiglu|veproj|baseline}"
            exit 1
            ;;
    esac
    echo "=== SUBMISSION COMPLETE ==="
}

# =========================================================================
# ABLATION PHASE 3: Validate SwiGLU + VE proj + WD 1.4 combo
# Architecture changes are now defaults in train.py
# =========================================================================
ablation3() {
    echo "=== ABLATION PHASE 3: SwiGLU + VE proj + WD 1.4 ==="

    # 1. New architecture with WD 1.4 (all defaults now)
    run_one "abl-combo-3ep" \
        --num-epochs=3 $BASELINE

    # 2. Compare: new architecture but with old WD 1.6
    run_one "abl-combo-wd16-3ep" \
        --num-epochs=3 --weight-decay=1.6 $BASELINE

    echo "=== ABLATION PHASE 3 COMPLETE ==="
}

# =========================================================================
# ABLATION PHASE 4: SwiGLU vs GeGLU vs VE proj, WD sweep
# Each tests one leaderboard-proven arch change + our WD improvement
# =========================================================================
ablation4() {
    echo "=== ABLATION PHASE 4 (3 epochs each) ==="

    # 1. SwiGLU + WD 1.4 (leaderboard #1 + our WD finding)
    run_one "swiglu-wd14-run1" \
        --num-epochs=3 --swiglu $BASELINE

    # 2. SwiGLU + WD 1.2 (push WD lower)
    run_one "swiglu-wd12-run1" \
        --num-epochs=3 --swiglu --weight-decay=1.2 $BASELINE

    # 3. VE proj + WD 1.4 (leaderboard #2 + our WD, properly with Muon)
    run_one "veproj-wd14-run1" \
        --num-epochs=3 --ve-proj $BASELINE

    # 4. SwiGLU + WD 1.4 run 2 (variance check)
    run_one "swiglu-wd14-run2" \
        --num-epochs=3 --swiglu $BASELINE

    # 5. SwiGLU + WD 1.6 (control — matches SwiGLU PR exactly)
    run_one "swiglu-wd16-run1" \
        --num-epochs=3 --swiglu --weight-decay=1.6 $BASELINE

    # 6. GeGLU + WD 1.4 (GELU gating variant)
    run_one "geglu-wd14-run1" \
        --num-epochs=3 --swiglu --geglu $BASELINE

    echo "=== ABLATION PHASE 4 COMPLETE ==="
}

# =========================================================================
# ABLATION PHASE 5: Fixed combo (SwiGLU + VE proj with Muon)
# =========================================================================
ablation5() {
    echo "=== ABLATION PHASE 5: Fixed combo ==="

    run_one "combo-swiglu-veproj-wd12" \
        --num-epochs=3 --swiglu --ve-proj --weight-decay=1.2 $BASELINE

    echo "=== ABLATION PHASE 5 COMPLETE ==="
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
    ablation)  ablation ;;
    ablation2) ablation2 ;;
    ablation3) ablation3 ;;
    ablation4) ablation4 ;;
    ablation5) ablation5 ;;
    submit)    submit ;;
    single)    shift; single "$@" ;;
    *)         echo "Usage: $0 {ablation|ablation2|ablation3|ablation4|submit|single <name> <args...>}" ;;
esac
