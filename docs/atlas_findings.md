# Atlas Architecture (arXiv 2505.23735) — JURECA Bring-up Findings

Date: 2026-04-01
System: JURECA dc-hwai (H100 80GB SXM, 93 GiB usable)

## Summary

The Atlas implementation in `nanochat/atlas.py` is functionally correct but has two blocking
issues that prevent practical training: an OOM from autograd activation storage, and extreme
Python-loop overhead that yields ~0.04% MFU.

## Issue 1: CUDA OOM in `polar_express` autograd graph

**Root cause:** `AtlasMemoryLayer.forward()` calls `polar_express()` once per chunk. Each call
runs 5 Newton-Schulz iterations, each producing multiple intermediate tensors of shape
`(B, chunk_size, H, D, D)`. Autograd retains all of these for backward.

With seq_len=2048, chunk_size=64 there are 32 chunks/layer x 12 layers = 384 calls.
Each call retains ~25 tensors. Even at B=1 the retained activations exceed 93 GiB.

**Evidence:**
- Job 14614557 (B=32): OOM trying to allocate 384 MiB, 91.58 GiB already allocated.
- Job 14614625 (B=8):  OOM trying to allocate 96 MiB (= 384/4, scales linearly with B),
  91.77 GiB already allocated. Total allocation barely changed — confirms the problem is
  the number of retained tensors, not batch size.

**Fix:** Gradient checkpointing (`torch.utils.checkpoint`) around each chunk's
`_process_chunk()` call (commit 6f727d7). Only M and S at chunk boundaries are stored;
intra-chunk intermediates are recomputed during backward.

**Result:** d4 smoke test (Job 14614758) completed 20 steps with peak memory 668 MiB.

## Issue 2: Near-zero MFU from uncompiled sequential Python loops

**Root cause:** `AtlasMemoryLayer.forward()` has two sequential Python for-loops per chunk
(momentum scan and memory scan), each iterating `chunk_size=64` times. These are true
recurrent dependencies (each step depends on the previous).

`torch.compile` cannot efficiently handle these loops — it attempts to unroll/trace all
64 x 2 x n_chunks x n_layers iterations, causing compilation to hang or take extremely long.
We skip `torch.compile` for Atlas (guard added in `base_train.py` line 258).

Without compilation:
- Each loop iteration launches separate small CUDA kernels
- Python interpreter overhead dominates (GIL, dispatch, etc.)
- No kernel fusion across iterations
- Gradient checkpointing doubles the loop count (recompute in backward)

**Evidence:**
- Job 14614805 (d8, 58M params, B=16, 8 grad_accum, FP8):
  - Step time: ~155s (vs ~1-2s for comparable GPT)
  - MFU: 0.04% (vs ~40-50% for GPT)
  - tok/sec: 1,688

## Issue 3: Slow initial validation eval

The default `--eval-every=250` triggers a validation eval at step 0, running 640 forward
passes of the uncompiled Atlas model. This takes tens of minutes and appears as a hang
before any training output is printed.

**Fix:** Pass `--eval-every=-1` for testing, or reduce `--eval-tokens`.

## What would be needed for practical Atlas training

1. **Fused recurrent scan kernel** (Triton or CUDA): replace the Python for-loops in the
   momentum and memory scans with a single fused kernel that runs the full scan on-GPU.
   This is how other recurrent architectures (Mamba, RWKV, etc.) achieve competitive MFU.

2. **Fused polar_express kernel**: the 5-step Newton-Schulz iteration on batched matrices
   could be a single Triton kernel instead of 5 separate matmul launches.

3. With fused kernels, `torch.compile` would likely work (no Python loops to trace),
   removing the need for the compile-skip guard.

## Files modified

- `scripts/base_train.py`: Skip `torch.compile` for Atlas model (lines 258-263)
- `nanochat/atlas.py`: Gradient checkpointing on `_process_chunk()` (commit 6f727d7)
- `runs/atlas_d12.slurm`: Various test configurations (not production-ready)
