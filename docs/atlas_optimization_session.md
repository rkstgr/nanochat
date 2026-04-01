# Atlas Performance Optimization Session — 2026-04-01

System: JURECA dc-hwai (H100 80GB SXM), login node (2× Quadro RTX 8000 46GB)

## Starting point

Atlas (arXiv 2505.23735) implementation was functionally correct but impractical:
- **155s/step**, 1,688 tok/sec, 0.04% MFU
- Root cause: two Python for-loops per chunk (64 iterations each) in momentum scan and memory scan, generating ~24M tiny CUDA kernel launches per training step
- `torch.compile` disabled for Atlas because it couldn't handle the loops

Config: d8, 58M params, B=16, 8 grad_accum, FP8, seq_len=2048, chunk_size=64, ns_steps=5, n_head=4, n_embd=512 (D=128)

## Optimization 1: Fused linear scan Triton kernel

**File**: `nanochat/atlas_kernels.py` — `fused_linear_scan()`

Replaced the Python for-loops `h_t = gate_t * h_{t-1} + input_t` with a single Triton kernel launch. Each program handles a BLOCK-sized slice of the D×D state vector for one (batch, head) pair, iterating sequentially over T=64 timesteps inside the kernel.

Custom `torch.autograd.Function` backward: the backward of the linear scan is itself a linear scan in reverse, so the same Triton kernel is reused on time-reversed tensors.

Both scans (momentum + memory) use this kernel.

**Result**: 155s → **44s/step** (3.5× speedup), 5,955 tok/sec

## Optimization 2: Fused Polar Express Triton kernel (forward only)

**File**: `nanochat/atlas_kernels.py` — `fused_polar_express()`

The 5 Newton-Schulz iterations involve 15 batched D×D matmuls. Each was a separate cuBLAS launch via PyTorch. The fused kernel runs all 5 iterations in a single Triton program per matrix, keeping the D×D state in registers via `tl.dot`.

Requires ≥128KB shared memory → works on H100 (228KB), fails on RTX 8000 (64KB) and falls back to PyTorch. D is padded to next power of 2 for `tl.arange` compatibility (D=96→128, D=128→128).

Forward-only: the backward still goes through autograd (30 cuBLAS matmul calls). Under gradient checkpointing, the fused forward runs twice (original + recompute), saving 30 of 60 total matmul launches.

**Result**: 44s → **19.5s/step** (2.25× additional), 13,418 tok/sec, 3.08% MFU

## Optimization 3: FLOP estimate fix + configurable params

The FLOP estimate used D² instead of D³ for the Newton-Schulz matmuls, underreporting by 8.6×. Fixed in `Atlas.estimate_flops()`. With the corrected estimate, MFU is reported correctly as ~3% (was showing 0.16%).

Added `--chunk-size` and `--ns-steps` CLI args to `base_train.py`.

## Optimization 4: torch.compile on _process_chunk

With Python loops eliminated, `torch.compile` can now trace `_process_chunk`. Compiling just this function (not the full model, which has the chunk loop) gives kernel fusion for element-wise ops and einsums. Minor improvement (~1.4× on RTX 8000, negligible on H100).

## Additional config: ns_steps=3

Reducing Newton-Schulz iterations from 5→3 gives nearly identical loss trajectory at this scale, with 40% less PE compute:

**Result with ns_steps=3**: **15.3s/step**, 17,185 tok/sec

## Approaches tested and rejected

### torch.compile(polar_express, mode="reduce-overhead")

CUDA graph capture eliminates kernel launch overhead for both forward and backward. 1.53× faster on isolated PE microbenchmark. But:
- `reduce-overhead` mode (CUDA graphs) is **incompatible with gradient checkpointing** — fails with "input tensor deallocate during graph recording"
- Default compile mode is 2.6× slower than current in full model context (replaces fast Triton forward with 15 compiled cuBLAS calls)

### Triton autograd.Function with fused backward kernel

The backward needs ~9 intermediate D×D matrices simultaneously in registers (grad, X, A, AA, B, dB, dA, dA_sym, temp). At D=128 in fp32: 9 × 128 × 128 × 4 = 589KB. **Exceeds H100's 228KB shared memory limit.** Infeasible without tiling, which would require a fundamentally different kernel design.

### gram-newton-schulz library (Dao-AILab)

Requires Python 3.12+ (we have 3.10). Custom symmetric GEMM kernels only activate for matrices ≥256×256 (our D=128). The Gram NS variant doesn't help for square matrices (G=X^TX is same size as X). The key insight (torch.compile with CUDA graphs) was tested independently above.

## Final results

| Configuration | Step time | tok/sec | Speedup | MFU |
|---|---|---|---|---|
| Baseline (Python loops, no compile) | 155s | 1,688 | 1× | 0.04% |
| + fused scan kernels | 44s | 5,955 | 3.5× | — |
| + fused PE forward kernel | 19.5s | 13,418 | 7.9× | 3.08% |
| + ns_steps=3 | 15.3s | 17,185 | 10.1× | — |

## Remaining bottleneck

89% of Atlas FLOPs are in Polar Express. The backward pass (30 cuBLAS matmuls per chunk via autograd) is the dominant remaining cost. Fusing it requires either:
- GPUs with >512KB shared memory per SM
- A tiled backward kernel design (complex, multiple tile-level matmul passes)
- An algorithmic change that reduces backward matmul count

Other potential improvements (lower priority):
- Larger chunk_size (fewer chunks → less overhead, better cuBLAS batching)
- Remove gradient checkpointing (memory permits: 24GB used of 80GB)
- Fuse M_t @ q_t into the memory scan kernel (avoids storing M_all)
- Fuse momentum_input = -η·u into the scan kernel

## Files modified

- `nanochat/atlas_kernels.py` — NEW: fused scan kernel, fused PE kernel, autograd Functions
- `nanochat/atlas.py` — use fused kernels, fix FLOP estimate
- `scripts/base_train.py` — add `--chunk-size`/`--ns-steps` args, compile `_process_chunk`
- `runs/atlas_fused_test.slurm`, `runs/atlas_fused_pe.slurm` — test job scripts

## Commit

`11cb687` — Fused Triton kernels for Atlas: 8x throughput improvement (155s → 19.5s/step)
