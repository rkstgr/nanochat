"""Test correctness and performance of polar express approaches."""
import torch
import time
import sys

from nanochat.atlas_kernels import (
    _polar_express_pytorch,
    fused_polar_express,
    compiled_polar_express,
    triton_autograd_polar_express,
)

device = 'cuda'
dtype = torch.float32
torch.manual_seed(42)

D = 64  # use 64 for RTX 8000 (128 needs H100 shared memory)
B = 256
steps = 5

print(f"Config: B={B}, D={D}, steps={steps}, device={torch.cuda.get_device_name()}")
print()

# === Correctness tests ===
X = torch.randn(B, D, D, device=device, dtype=dtype, requires_grad=True)

# Reference
ref = _polar_express_pytorch(X, steps)
loss_ref = ref.sum()
loss_ref.backward()
grad_ref = X.grad.clone()
X.grad = None

# Approach A: compiled
try:
    comp = compiled_polar_express(X, steps)
    loss_comp = comp.sum()
    loss_comp.backward()
    grad_comp = X.grad.clone()
    X.grad = None
    fwd_diff_a = (comp - ref).abs().max().item()
    bwd_diff_a = (grad_comp - grad_ref).abs().max().item()
    print(f"Approach A (compiled):       fwd diff={fwd_diff_a:.2e}, bwd diff={bwd_diff_a:.2e}",
          "PASS" if fwd_diff_a < 1e-3 and bwd_diff_a < 1e-3 else "FAIL")
except Exception as e:
    print(f"Approach A (compiled):       FAILED - {e}")

# Approach B: Triton autograd
try:
    tri = triton_autograd_polar_express(X, steps)
    loss_tri = tri.sum()
    loss_tri.backward()
    grad_tri = X.grad.clone()
    X.grad = None
    fwd_diff_b = (tri - ref).abs().max().item()
    bwd_diff_b = (grad_tri - grad_ref).abs().max().item()
    print(f"Approach B (triton autograd): fwd diff={fwd_diff_b:.2e}, bwd diff={bwd_diff_b:.2e}",
          "PASS" if fwd_diff_b < 1e-3 and bwd_diff_b < 1e-3 else "FAIL")
except Exception as e:
    print(f"Approach B (triton autograd): FAILED - {e}")

# Current: fused forward, autograd backward
try:
    fus = fused_polar_express(X, steps)
    loss_fus = fus.sum()
    loss_fus.backward()
    grad_fus = X.grad.clone()
    X.grad = None
    fwd_diff_c = (fus - ref).abs().max().item()
    bwd_diff_c = (grad_fus - grad_ref).abs().max().item()
    print(f"Current (fused fwd only):    fwd diff={fwd_diff_c:.2e}, bwd diff={bwd_diff_c:.2e}",
          "PASS" if fwd_diff_c < 1e-3 and bwd_diff_c < 1e-3 else "FAIL")
except Exception as e:
    print(f"Current (fused fwd only):    FAILED - {e}")

# === Performance tests (forward + backward) ===
print("\n=== Performance (forward + backward) ===")

def bench(name, fn, N=30, warmup=10):
    X = torch.randn(B, D, D, device=device, dtype=dtype, requires_grad=True)
    for _ in range(warmup):
        y = fn(X, steps)
        y.sum().backward()
        X.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        y = fn(X, steps)
        y.sum().backward()
        X.grad = None
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"{name:40s} {ms:8.2f} ms")
    return ms

t_ref = bench("Reference (PyTorch)", _polar_express_pytorch)
try:
    t_a = bench("Approach A (compiled reduce-overhead)", compiled_polar_express)
except Exception as e:
    print(f"Approach A benchmark failed: {e}")
    t_a = float('inf')
try:
    t_b = bench("Approach B (triton autograd fwd+bwd)", triton_autograd_polar_express)
except Exception as e:
    print(f"Approach B benchmark failed: {e}")
    t_b = float('inf')

print(f"\nSpeedups over reference:")
if t_a < float('inf'):
    print(f"  Approach A: {t_ref/t_a:.2f}x")
if t_b < float('inf'):
    print(f"  Approach B: {t_ref/t_b:.2f}x")
if t_a < float('inf') and t_b < float('inf'):
    winner = "A (compiled)" if t_a < t_b else "B (triton autograd)"
    print(f"  Winner: {winner} ({min(t_a,t_b):.2f} ms vs {max(t_a,t_b):.2f} ms)")
