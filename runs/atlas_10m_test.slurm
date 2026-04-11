#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=atlas-10m-test
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas (atlas-pytorch AtlasLMM) ~10M test run, 1×H100
# depth=6, dim=128, 2 heads — ~10M params
# ~120M tokens, 500 steps, ~10 min
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

export TORCH_EXTENSIONS_DIR="/p/scratch/westai0047/nanochat/torch_extensions"
export TRITON_CACHE_DIR="/p/scratch/westai0047/nanochat/triton_cache"
export XDG_CACHE_HOME="/p/scratch/westai0047/nanochat/cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

echo "==> Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "==> Start time: $(date)"

cd "$REPO_DIR"
source .venv/bin/activate

# 1 GPU: total_batch_size = 32 * 2048 = 65536 (no grad accum needed)
# 120M / 65536 ~= 1831 tokens/step => but let's do 500 steps for quick test
# 500 * 65536 = 32M tokens
python -m scripts.base_train \
    --model=atlas \
    --depth=6 \
    --aspect-ratio=21 \
    --head-dim=64 \
    --max-seq-len=2048 \
    --omega-window=2 \
    --poly-degree=0 \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations=120 \
    --eval-every=40 \
    --eval-tokens=2097152 \
    --sample-every=250 \
    --core-metric-every=-1 \
    --save-every=-1 \
    --warmup-steps=20 \
    --warmdown-ratio=0.65 \
    --run="atlas-10m-test" \
    --model-tag="atlas-10m-test"

echo "==> End time: $(date)"
