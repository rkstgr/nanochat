"""Test fused polar_express with D=128."""
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
def _pe_kernel_tiled(
    X_ptr, OUT_ptr, coeffs_ptr,
    batch_stride,
    NS_STEPS: tl.constexpr,
    D: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fused Polar Express with tiled matmuls for larger D.
    Each program handles one D×D matrix. Matmuls are tiled with TILE×TILE blocks."""
    bid = tl.program_id(0)
    base = X_ptr + bid * batch_stride
    out_base = OUT_ptr + bid * batch_stride

    n_tiles = D // TILE

    # Load full D×D matrix into registers, tile by tile
    # We'll use global memory as our "register file" — read/write tiles as needed
    # For the fused kernel, intermediates (A, B) go to a scratch buffer
    # Actually, let's just use output buffer as scratch since we write final result last

    # For simplicity with large D, we'll use global memory for intermediate storage
    # but fuse all 5 NS steps into one kernel to avoid launch overhead.
    # The memory traffic per step is 3 matmuls × 2 reads + 1 write.

    # Scratch space: reuse output buffer for A, and a portion for AA
    # Actually this gets messy. Let me just do the naive approach:
    # load X, compute everything, store X. Use tiled matmul with accumulation.

    # We need to store the full D×D matrices X, A, B somewhere.
    # With TILE=64, D=128, we have 4 tiles per matrix.

    # Strategy: Keep X in global memory at X_ptr[bid].
    # Write A to OUT_ptr[bid] (we'll overwrite with final X at the end).
    # We need one more scratch buffer... let's use the approach of computing
    # each output tile on-the-fly.

    # Actually, the simplest approach for large D: just do the 5 NS steps
    # using global memory reads/writes, but in a SINGLE kernel launch.
    # This saves 14 kernel launches (5 steps × 3 matmuls - 1 initial = 14 extra launches).

    X_base = base  # X lives at input location
    # We'll use a small scratch area within the same matrix space
    # Hmm, we can't easily get scratch without extra memory.

    # Let's just read/write X in place and use OUT as scratch for A.
    # Step pattern:
    # 1. Normalize X in-place (read X, write normalized X back to X_base)
    # 2. For each NS step:
    #    a. Compute A = X @ X.T -> store to OUT buffer
    #    b. Compute AA = A @ A -> need another scratch...
    #       We can compute B = b*A + c*(A@A) tile by tile without storing AA fully:
    #       B[i,j] = b*A[i,j] + c*sum_k A[i,k]*A[k,j]
    #       This is just b*A + c*A@A. We need A fully computed first.
    #       We can compute B[i_tile, j_tile] = b*A[i_tile, j_tile] + c * (A@A)[i_tile, j_tile]
    #       where (A@A)[i_tile, j_tile] = sum_k A[i_tile, k_tile] @ A[k_tile, j_tile]
    #       This requires reading all of A (from OUT buffer) for the tiled matmul.
    #    c. Compute X_new = a*X + B@X

    # This works! OUT buffer holds A, and we read it for AA and BX computations.

    A_base = out_base  # Use output buffer to store A temporarily

    # --- Frobenius normalize ---
    norm_sq = tl.zeros((), dtype=tl.float32)
    for ti in range(n_tiles):
        for tj in range(n_tiles):
            r = tl.arange(0, TILE) + ti * TILE
            c = tl.arange(0, TILE) + tj * TILE
            tile = tl.load(X_base + r[:, None] * D + c[None, :]).to(tl.float32)
            norm_sq += tl.sum(tile * tile)

    inv_norm = 1.0 / (tl.sqrt(norm_sq) * 1.01 + 1e-6)

    for ti in range(n_tiles):
        for tj in range(n_tiles):
            r = tl.arange(0, TILE) + ti * TILE
            c = tl.arange(0, TILE) + tj * TILE
            tile = tl.load(X_base + r[:, None] * D + c[None, :]).to(tl.float32)
            tile = tile * inv_norm
            tl.store(X_base + r[:, None] * D + c[None, :], tile)

    # --- 5 Newton-Schulz steps ---
    for step in range(NS_STEPS):
        a_coeff = tl.load(coeffs_ptr + step * 3).to(tl.float32)
        b_coeff = tl.load(coeffs_ptr + step * 3 + 1).to(tl.float32)
        c_coeff = tl.load(coeffs_ptr + step * 3 + 2).to(tl.float32)

        # A = X @ X.T -> store to A_base
        for ti in range(n_tiles):
            for tj in range(n_tiles):
                acc = tl.zeros((TILE, TILE), dtype=tl.float32)
                for tk in range(n_tiles):
                    r = tl.arange(0, TILE) + ti * TILE
                    k1 = tl.arange(0, TILE) + tk * TILE
                    x_ik = tl.load(X_base + r[:, None] * D + k1[None, :]).to(tl.float32)

                    c_idx = tl.arange(0, TILE) + tj * TILE
                    x_jk = tl.load(X_base + c_idx[:, None] * D + k1[None, :]).to(tl.float32)

                    acc += tl.dot(x_ik, tl.trans(x_jk))

                r = tl.arange(0, TILE) + ti * TILE
                c_idx = tl.arange(0, TILE) + tj * TILE
                tl.store(A_base + r[:, None] * D + c_idx[None, :], acc)

        # X_new = a*X + (b*A + c*(A@A)) @ X
        # Compute per output tile of X_new:
        for ti in range(n_tiles):
            for tj in range(n_tiles):
                # Accumulate: sum_k B[ti,tk] @ X[tk,tj] where B = b*A + c*(A@A)
                acc = tl.zeros((TILE, TILE), dtype=tl.float32)
                for tk in range(n_tiles):
                    # Load A[ti, tk]
                    r = tl.arange(0, TILE) + ti * TILE
                    k1 = tl.arange(0, TILE) + tk * TILE
                    a_tile = tl.load(A_base + r[:, None] * D + k1[None, :]).to(tl.float32)

                    # Compute B[ti, tk] = b*A[ti,tk] + c*(A@A)[ti,tk]
                    # (A@A)[ti,tk] = sum_m A[ti,m] @ A[m,tk]
                    aa_tile = tl.zeros((TILE, TILE), dtype=tl.float32)
                    for tm in range(n_tiles):
                        m = tl.arange(0, TILE) + tm * TILE
                        a_im = tl.load(A_base + r[:, None] * D + m[None, :]).to(tl.float32)
                        a_mk = tl.load(A_base + m[:, None] * D + k1[None, :]).to(tl.float32)
                        aa_tile += tl.dot(a_im, a_mk)

                    b_tile = b_coeff * a_tile + c_coeff * aa_tile

                    # Load X[tk, tj]
                    c_idx = tl.arange(0, TILE) + tj * TILE
                    x_kj = tl.load(X_base + k1[:, None] * D + c_idx[None, :]).to(tl.float32)

                    acc += tl.dot(b_tile, x_kj)

                # Load X[ti, tj] for a*X term
                r = tl.arange(0, TILE) + ti * TILE
                c_idx = tl.arange(0, TILE) + tj * TILE
                x_ij = tl.load(X_base + r[:, None] * D + c_idx[None, :]).to(tl.float32)

                x_new = a_coeff * x_ij + acc
                tl.store(X_base + r[:, None] * D + c_idx[None, :], x_new)

    # Result is stored in X_base (modified in-place)


def polar_express_fused_tiled(X, steps=5, tile=64):
    shape = X.shape
    D = shape[-1]
    assert D % tile == 0, f"D={D} must be divisible by tile={tile}"
    X_flat = X.reshape(-1, D, D).contiguous().clone()  # clone because we modify in-place
    N = X_flat.shape[0]
    OUT = torch.empty_like(X_flat)

    coeffs = torch.tensor(
        [v for abc in _COEFFS[:steps] for v in abc],
        device=X.device, dtype=torch.float32
    )

    _pe_kernel_tiled[(N,)](
        X_flat, OUT, coeffs,
        batch_stride=D * D,
        NS_STEPS=steps,
        D=D,
        TILE=tile,
    )
    # Result is in X_flat (modified in-place), not OUT (used as scratch for A)
    return X_flat.reshape(shape)


# Test correctness
device = 'cuda'
dtype = torch.float32

for D in [64, 128]:
    X = torch.randn(32, D, D, device=device, dtype=dtype)
    ref = polar_express_ref(X)
    fused = polar_express_fused_tiled(X, tile=min(64, D))
    diff = (ref - fused).abs().max().item()
    print(f"D={D}: max diff = {diff:.2e} {'PASS' if diff < 1e-3 else 'FAIL'}")

# Benchmark
B, T, H, D = 16, 64, 4, 128  # H100 config: n_head=4, D=128
X = torch.randn(B * T * H, D, D, device=device, dtype=dtype)

for _ in range(5):
    polar_express_ref(X)
    polar_express_fused_tiled(X, tile=64)
torch.cuda.synchronize()

N = 50
t0 = time.perf_counter()
for _ in range(N):
    polar_express_ref(X)
torch.cuda.synchronize()
ref_ms = (time.perf_counter() - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N):
    polar_express_fused_tiled(X, tile=64)
torch.cuda.synchronize()
fused_ms = (time.perf_counter() - t0) / N * 1000

print(f"\nBatch={B*T*H}, D={D}")
print(f"Reference:  {ref_ms:.2f} ms")
print(f"Fused:      {fused_ms:.2f} ms")
print(f"Speedup:    {ref_ms/fused_ms:.2f}x")
