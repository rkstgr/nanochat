# Atlas Testing & Performance Plan — 16×H100 Target

## Context

Three paper features (arXiv 2505.23735) were added to `nanochat/atlas.py`: deep MLP memory, Omega rule (sliding window gradients), and polynomial feature mapping. None had tests. The deep memory path doubles kernel launches (6 per chunk vs 3) and wasn't torch.compiled. Current MFU on H100 is 3.08% with the linear memory path — indicating severe launch-boundedness, not compute-boundedness. The PE backward (autograd, 30 cuBLAS calls/chunk) is 89% of FLOPs. Targeting efficient training on up to 16×H100 (2 nodes × 8 GPUs).

## Critical files

| File | Role |
|------|------|
| `nanochat/atlas.py` | Model: memory layer, chunk processing, state management |
| `nanochat/atlas_kernels.py` | Triton kernels: fused scan, fused PE, PE STE |
| `scripts/base_train.py` | Training loop, torch.compile, DDP, FP8 |
| `nanochat/optim.py` | MuonAdamW / DistMuonAdamW |
| `tests/test_atlas_features.py` | Feature correctness tests (omega, poly, deep memory) |
| `tests/test_atlas_gradients.py` | Gradient correctness tests (gradcheck) |
| `tests/atlas/` | Existing PE kernel tests (6 files) |

---

## Part 1: Testing

### 1A. Feature correctness — `tests/test_atlas_features.py` ✅ DONE

Small configs (D=16 or 32, H=2, chunk_size=8) on CPU for fast pytest runs.

- **Omega rule**: Compare `_omega_aggregate(u, gamma, w)` against a naive double-loop reference. Test w=1 (identity), w=chunk_size (full sum), w=3 (partial). Test gamma=0 masks contributions.
- **Polynomial features**: `_poly_features` with poly_degree=1 returns a_1*x. poly_degree=3 matches manual a_1*x + a_2*x² + a_3*x³. Verify initial coefficients are [1, 0.5, 1/6]. Verify gradient flows through poly_coeffs.
- **Deep MLP memory**: Verify W2 identity init gives M(k)=k+0 when W1=0 but GELU(I@k)≠0. Verify output shape matches linear path. Verify memory_expand=2 produces (D,2D) and (2D,D) weight matrices. Test state persistence across chunks (W1/W2 evolve, not reset).
- **Config backward compat**: `AtlasConfig(omega_window=1, poly_degree=0, deep_memory=False)` produces identical forward output as the old code path.

Status: 12 CPU tests passing, 6 GPU-only tests written (skip on CPU).

### 1B. Gradient correctness — `tests/test_atlas_gradients.py` ✅ DONE

Use fp64 with tiny dims (D=4, H=1, chunk_size=4, seq_len=8) for `torch.autograd.gradcheck`.

- `_process_chunk` with omega_window=1 and omega_window=3
- `_process_chunk_deep` with memory_expand=1 (square, uses Triton on GPU)
- `_process_chunk_deep` with memory_expand=2 (rectangular, PyTorch PE fallback)
- `fused_linear_scan` in isolation (forward + backward)
- `_gelu_derivative` vs autograd-computed derivative
- Polynomial features gradcheck (coefficients + input)
- Omega aggregate gradcheck (u + gamma)

Status: 5 CPU tests passing, 5 GPU-only tests written (skip on CPU).

### 1C. Numerical stability — `tests/test_atlas_stability.py` — TODO

Requires GPU (mark `@pytest.mark.slow`).

- bf16 vs fp32 forward agreement (relative tolerance ~1e-2 for bf16)
- Long sequence (8192 tokens, 128 chunks) NaN/Inf check for both memory paths
- Polar Express output convergence: verify `PE(X) @ PE(X).T ≈ I` within tolerance
- Cumsum precision: `_omega_aggregate` at position 2048 doesn't drift vs chunked computation

### 1D. Integration — `tests/test_atlas_integration.py` — TODO

Short training run (mark `@pytest.mark.slow`), requires GPU.

- 20 steps of training with deep_memory=True: loss decreases monotonically after step 5
- 20 steps with deep_memory=False (linear): same check, compare convergence rate
- `Atlas.generate()` produces valid token IDs (not NaN/negative) after 10 training steps
- Gradient accumulation: 1×B=8 matches 2×B=4 loss within tolerance

### 1E. Distributed — `tests/test_atlas_distributed.py` — TODO

Requires multi-GPU (mark `@pytest.mark.slow`).

- Launch via `torch.multiprocessing.spawn(nprocs=2)` or `torchrun`
- Single-GPU loss vs 2-GPU DDP loss match within bf16 tolerance after 5 steps
- DistMuonAdamW gradient sync: verify all ranks have identical parameters after step

---

## Part 2: Performance Optimization

### Phase 0: Profile and Baseline ✅ PARTIALLY DONE

**Goal**: Confirm bottleneck structure for the new deep memory path.

1. ✅ **Wire new config into CLI** — `--omega-window`, `--poly-degree`, `--deep-memory`, `--memory-expand`, `--pe-ste` added to `scripts/base_train.py`
2. ✅ **torch.compile `_process_chunk_deep`** — added in `base_train.py` alongside existing `_process_chunk` compile
3. **Profile on H100** — TODO: Run `torch.profiler` with `--depth=8 --deep-memory=1` for 10 steps. Capture:
   - Time split: forward vs backward vs optimizer
   - Kernel launch count per step
   - PE backward fraction (expect >89% since deep memory doubles PE calls)
   - Memory high-water mark with/without gradient checkpointing
4. **Memory budget calculation**: With deep memory (memory_expand=1), 4 state matrices per head per layer:
   - Per sample: 4 × n_layer × n_head × D × D × 2 bytes (bf16)
   - d24, H=16, D=96: 4 × 24 × 16 × 96 × 96 × 2 = 27 MB/sample → B=32 uses 864 MB (trivial)
   - Activations during chunk processing dominate — profile to get actual number

**Profiling commands:**
```bash
# Baseline deep memory
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --model=atlas --depth=8 --deep-memory=1 --device-batch-size=16 \
    --num-iterations=10 --eval-every=-1 --run=atlas_deep_baseline

# Compare linear memory
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --model=atlas --depth=8 --deep-memory=0 --omega-window=1 --poly-degree=0 \
    --device-batch-size=16 --num-iterations=10 --eval-every=-1 --run=atlas_linear_baseline
```

### Phase 1: PE Straight-Through Estimator (highest impact) ✅ IMPLEMENTED, NEEDS VALIDATION

**Goal**: Eliminate 89%+ of backward FLOPs.

The Polar Express is an *internal optimizer step* (finding the nearest orthogonal matrix to the momentum), not a learned function. Its Jacobian near orthogonal matrices is approximately identity. We can use a straight-through estimator: full NS iteration in forward, pass gradients through unchanged in backward.

**Implementation** in `nanochat/atlas_kernels.py`:
```python
class _PolarExpressSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, steps):
        return fused_polar_express(X, steps)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through

def polar_express_ste(X, steps=5):
    return _PolarExpressSTE.apply(X, steps)
```

Wired into both `_process_chunk` and `_process_chunk_deep` via `use_pe_ste` parameter. Controlled by `AtlasConfig(pe_ste=True)` / `--pe-ste=1`.

**Validation** — TODO on cluster: Run a 1000-step ablation comparing loss curves with and without STE. If loss diverges, try a softer version: `grad_output * (1-β) + PE_true_grad * β` with small β.

```bash
# With STE
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --model=atlas --depth=8 --deep-memory=1 --pe-ste=1 --device-batch-size=16 \
    --num-iterations=1000 --eval-every=100 --run=atlas_ste_on

# Without STE (baseline)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train \
    --model=atlas --depth=8 --deep-memory=1 --pe-ste=0 --device-batch-size=16 \
    --num-iterations=1000 --eval-every=100 --run=atlas_ste_off
```

**Expected impact**: 4-5× speedup on backward pass. Combined with torch.compile, expect 3-4× total step time reduction.

### Phase 2: Reduce Kernel Launch Overhead — TODO

**Goal**: Cut kernel launches from ~4608/forward (deep path) by 2-4×.

**2a. Remove gradient checkpointing (if memory permits)**

From Phase 0 profiling, if peak memory with checkpointing disabled fits in 80GB for target batch size:
- Remove `checkpoint()` wrapper in `AtlasMemoryLayer.forward()`
- Add config flag `use_checkpoint: bool = True` for flexibility
- Halves total kernel launches (no backward recompute)
- **Expected**: 1.5-2× speedup if launch-bound

**2b. Fuse η-scaling into scan kernel**

Currently: `momentum_input = -(e_c.unsqueeze(-1) * u)` is a separate element-wise kernel, then `fused_linear_scan(S, theta, momentum_input)`. Fuse the η-multiply into the scan kernel:

```python
# In _scan_fwd_kernel: state = gate * state + eta * inp  (add eta parameter)
```

Saves 2 kernel launches per chunk (1 for W1 path, 1 for W2 path) = 64 per layer × 24 layers = 1536 fewer launches. The kernel change is straightforward — add a per-timestep scalar parameter.

**Files**: `nanochat/atlas_kernels.py` (kernel + autograd), `nanochat/atlas.py` (call site)

**2c. Parallel W1/W2 streams**

The W1 and W2 scan paths in `_process_chunk_deep` are independent after gradient computation. Run them on separate CUDA streams:

```python
with torch.cuda.stream(stream_w1):
    chunk_S_W1, S_W1 = fused_linear_scan(...)
    ...
with torch.cuda.stream(stream_w2):
    chunk_S_W2, S_W2 = fused_linear_scan(...)
    ...
torch.cuda.current_stream().wait_stream(stream_w1)
torch.cuda.current_stream().wait_stream(stream_w2)
```

This overlaps the two independent scan+PE+scan chains. May interact with torch.compile — test carefully.

**Files**: `nanochat/atlas.py`

### Phase 3: Triton Kernel Improvements — TODO

**3a. Rectangular PE kernel** (only needed if `memory_expand > 1`)

Currently falls back to PyTorch (15 separate cuBLAS calls). Write a Triton kernel handling M×N matrices:

```python
@triton.jit
def _polar_express_fwd_kernel_rect(X_ptr, OUT_ptr, coeffs_ptr,
    batch_stride, NS_STEPS, ROWS, COLS, PAD_ROWS, PAD_COLS):
    # A = X @ X^T: (ROWS, COLS) @ (COLS, ROWS) = (ROWS, ROWS)
    # B @ X: (ROWS, ROWS) @ (ROWS, COLS) = (ROWS, COLS)
```

Only implement if ablation shows `memory_expand=2` beats `memory_expand=1` on perplexity.

**3b. Larger chunk_size**

From the optimization doc: "Larger chunk_size → fewer chunks → less overhead, better cuBLAS batching." Test chunk_size=128 and 256. Each doubling halves kernel launches. Trade-off: omega_window must be ≤ chunk_size (already enforced), and activation memory per chunk increases.

**Files**: `nanochat/atlas_kernels.py`, config

### Phase 4: Multi-Node Scaling (16×H100) — TODO

**Goal**: Linear scaling from 8→16 GPUs across 2 nodes.

**4a. NCCL configuration**

On JURECA with InfiniBand:
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
```

Launch with:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 -m scripts.base_train ...
```

**4b. Communication-computation overlap**

DistMuonAdamW already uses async reduce_scatter + all_gather. Verify the async phases fully overlap with compute on 2 nodes. Profile NCCL time fraction.

**4c. Gradient compression** (if comm-bound)

For bf16 gradients over IB, bandwidth may be sufficient. If profiling shows >20% time in NCCL:
- Use `torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook`
- Or bf16 gradient compression via `register_comm_hook`

Atlas model is relatively small (58M-760M params), so communication should NOT be the bottleneck. The real scaling concern is maintaining high utilization per GPU.

**4d. Batch size scaling**

With 2× GPUs, either:
- Keep per-GPU batch size → 2× total batch → 2× learning rate (linear scaling rule)
- Halve gradient accumulation steps → same total batch, 2× throughput

**Files**: `scripts/base_train.py`, SLURM scripts in `runs/`

---

## Execution Order

| Priority | Task | Effort | Expected Impact | Status |
|----------|------|--------|-----------------|--------|
| **P0** | Wire CLI args + torch.compile `_process_chunk_deep` | 30 min | 10-30% (removes compile gap) | ✅ Done |
| **P0** | All correctness tests (1A, 1B) | 1 day | Risk reduction | ✅ Done (17 CPU passing, 11 GPU pending) |
| **P0** | PE straight-through estimator implementation | 1 hour | — | ✅ Done |
| **P1** | Run GPU tests on H100 | 30 min | Validates deep memory + kernels | TODO |
| **P1** | Profile deep memory on H100 | 2 hours | Informs all other decisions | TODO |
| **P1** | PE STE 1000-step ablation | 1 day | **3-5× backward speedup** | TODO |
| **P2** | Remove gradient checkpointing (if fits) | 1 hour | 1.5-2× (halves launches) | TODO |
| **P2** | Fuse η into scan kernel | 4 hours | 10-15% fewer launches | TODO |
| **P2** | Stability + integration tests (1C, 1D) | 1 day | Quality gate | TODO |
| **P3** | Parallel W1/W2 streams | 4 hours | Up to 1.5× on deep path | TODO |
| **P3** | Larger chunk_size experiments | 2 hours | Depends on memory | TODO |
| **P3** | Multi-node SLURM + NCCL tuning | 4 hours | Linear 8→16 scaling | TODO |
| **P4** | Rectangular PE kernel | 1 day | Only if expand>1 needed | TODO |
| **P4** | Distributed tests (1E) | 4 hours | Multi-node safety | TODO |

## Verification

1. **Correctness**: `uv run --group dev pytest tests/test_atlas_features.py tests/test_atlas_gradients.py -v` passes (all 28 tests)
2. **Stability**: `uv run --group dev pytest tests/test_atlas_stability.py tests/test_atlas_integration.py -v --slow` passes on GPU
3. **Performance**: Compare step time and MFU before/after each optimization on H100 with `--model=atlas --depth=8 --deep-memory=1`
4. **STE validation**: Loss curve over 1000 steps with `--pe-ste=1` matches `--pe-ste=0` within 5% at convergence
5. **Multi-node**: `torchrun --nnodes=2 --nproc_per_node=8` shows >90% scaling efficiency vs single-node
6. **Regression**: Linear memory path (`--deep-memory=0 --omega-window=1 --poly-degree=0`) performance unchanged

## Config Reference

```python
AtlasConfig(
    omega_window=16,    # Omega rule window (1=online Delta rule, paper default=16)
    poly_degree=3,      # Polynomial features (0=disabled, paper default=3)
    deep_memory=True,   # Deep MLP memory vs linear matrix
    memory_expand=1,    # MLP expansion (1=D×D, 2=D×2D — needs PyTorch PE fallback)
    pe_ste=False,       # PE straight-through estimator (EXPERIMENTAL — validate first)
)
```

CLI equivalents: `--omega-window=16 --poly-degree=3 --deep-memory=1 --memory-expand=1 --pe-ste=0`
