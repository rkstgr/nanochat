#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=atlas_fused
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas d8 test: fused Triton scans + torch.compile on 1×H100
# Compare against baseline Job 14614805 (155s/step, 0.04% MFU)
# ==============================================================================

set -euo pipefail

module load uv

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO_DIR="/p/project1/westai0047/nanochat"
export NANOCHAT_BASE_DIR="/p/scratch/westai0047/nanochat"

export HF_HOME="/p/scratch/westai0047/huggingface"
export HF_DATASETS_CACHE="/p/scratch/westai0047/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

WANDB_OUT="/p/scratch/westai0047/nanochat/wandb"
mkdir -p "$WANDB_OUT"
export WANDB_MODE=offline
export WANDB_DIR="$WANDB_OUT"

echo "==> Nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes)"
echo "==> Start time: $(date)"

cd "$REPO_DIR"
source .venv/bin/activate

TORCHRUN="srun torchrun --standalone --nproc_per_node=1"

echo "==> Starting Atlas d8 with fused Triton scans + torch.compile..."
$TORCHRUN -m scripts.base_train -- \
    --model=atlas \
    --depth=8 \
    --run="atlas-d8-fused" \
    --model-tag="atlas-d8-fused" \
    --device-batch-size=16 \
    --num-iterations=30 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --eval-every=-1 \
    --fp8

echo "==> Training complete at $(date)"
