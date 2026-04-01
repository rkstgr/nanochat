"""Test simple (non-tiled) polar_express kernel with D=128."""
import torch
import triton
import triton.language as tl
import time

_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def polar_express_ref(X, steps=5):
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in _COEFFS[:steps]:
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


@triton.jit
def _pe_kernel_simple(
    X_ptr, OUT_ptr, coeffs_ptr,
    batch_stride,
    NS_STEPS: tl.constexpr,
    D: tl.constexpr,
):
    """Fused Polar Express: one program per matrix, full D×D via tl.dot."""
    bid = tl.program_id(0)
    base = X_ptr + bid * batch_stride
    out_base = OUT_ptr + bid * batch_stride

    rows = tl.arange(0, D)
    cols = tl.arange(0, D)
    idx = rows[:, None] * D + cols[None, :]

    X = tl.load(base + idx).to(tl.float32)

    norm = tl.sqrt(tl.sum(X * X))
    X = X / (norm * 1.01 + 1e-6)

    for step in range(NS_STEPS):
        a = tl.load(coeffs_ptr + step * 3).to(tl.float32)
        b = tl.load(coeffs_ptr + step * 3 + 1).to(tl.float32)
        c = tl.load(coeffs_ptr + step * 3 + 2).to(tl.float32)

        A = tl.dot(X, tl.trans(X))
        AA = tl.dot(A, A)
        B = b * A + c * AA
        X = a * X + tl.dot(B, X)

    tl.store(out_base + idx, X)


def polar_express_fused(X, steps=5):
    shape = X.shape
    D = shape[-1]
    X_flat = X.reshape(-1, D, D).contiguous()
    N = X_flat.shape[0]
    OUT = torch.empty_like(X_flat)

    coeffs = torch.tensor(
        [v for abc in _COEFFS[:steps] for v in abc],
        device=X.device, dtype=torch.float32
    )

    _pe_kernel_simple[(N,)](
        X_flat, OUT, coeffs,
        batch_stride=D * D,
        NS_STEPS=steps,
        D=D,
    )
    return OUT.reshape(shape)


device = 'cuda'
dtype = torch.float32

for D in [64, 96, 128]:
    try:
        X = torch.randn(32, D, D, device=device, dtype=dtype)
        ref = polar_express_ref(X)
        fused = polar_express_fused(X)
        diff = (ref - fused).abs().max().item()
        print(f"D={D}: max diff = {diff:.2e} {'PASS' if diff < 1e-3 else 'FAIL'}")
    except Exception as e:
        print(f"D={D}: FAILED - {e}")

# Benchmark with D=128 if it works
D = 128
batch = 4096
try:
    X = torch.randn(batch, D, D, device=device, dtype=dtype)
    for _ in range(3): polar_express_ref(X); polar_express_fused(X)
    torch.cuda.synchronize()
    N = 50
    t0 = time.perf_counter()
    for _ in range(N): polar_express_ref(X)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / N * 1000
    t0 = time.perf_counter()
    for _ in range(N): polar_express_fused(X)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / N * 1000
    print(f"\nBatch={batch}, D={D}")
    print(f"Reference: {ref_ms:.2f} ms, Fused: {fused_ms:.2f} ms, Speedup: {ref_ms/fused_ms:.2f}x")
except Exception as e:
    print(f"\nD={D} benchmark failed: {e}")
