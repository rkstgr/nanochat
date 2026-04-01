"""Fused Triton kernels for Atlas linear scans.

The Atlas memory layer has two sequential scans per chunk:
1. Momentum scan: S_t = theta_t * S_{t-1} - eta_t * u_t
2. Memory scan:   M_t = alpha_t * M_{t-1} + S'_t

Both are linear recurrences: h_t = gate_t * h_{t-1} + input_t
where gate_t is a scalar per (batch, head) and h_t is a (D, D) matrix.

The Python for-loops over chunk_size (default 64) timesteps generate millions
of tiny CUDA kernel launches, causing extreme dispatch overhead (~155s/step).
This module fuses each full scan into a single Triton kernel launch, reducing
kernel launch count by ~1000x.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_fwd_kernel(
    h_init_ptr, gates_ptr, inputs_ptr, h_all_ptr,
    T,
    # h_init: (B, H, DD) — strides for dims 0, 1
    s_hi_0, s_hi_1,
    # gates: (B, T, H) — strides for dims 0, 1, 2
    s_g_0, s_g_1, s_g_2,
    # inputs: (B, T, H, DD) — strides for dims 0, 1, 2
    s_i_0, s_i_1, s_i_2,
    # h_all: (B, T, H, DD) — strides for dims 0, 1, 2
    s_o_0, s_o_1, s_o_2,
    H: tl.constexpr,
    DD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward linear scan: h_t = gate_t * h_{t-1} + input_t, t=0..T-1.

    Each program handles BLOCK elements of the DD-dimensional state vector
    for one (batch, head) pair, iterating sequentially over T timesteps.
    """
    pid = tl.program_id(0)
    num_dd_blocks = tl.cdiv(DD, BLOCK)
    bh = pid // num_dd_blocks
    dd_block = pid % num_dd_blocks
    b = bh // H
    h = bh % H

    offs = dd_block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DD

    # Load initial state: h_init[b, h, offs]
    state = tl.load(h_init_ptr + b * s_hi_0 + h * s_hi_1 + offs, mask=mask, other=0.0)

    for t in range(T):
        # gate: scalar gates[b, t, h]
        gate = tl.load(gates_ptr + b * s_g_0 + t * s_g_1 + h * s_g_2)
        # input: inputs[b, t, h, offs]
        inp = tl.load(inputs_ptr + b * s_i_0 + t * s_i_1 + h * s_i_2 + offs, mask=mask, other=0.0)
        state = gate * state + inp
        # store: h_all[b, t, h, offs]
        tl.store(h_all_ptr + b * s_o_0 + t * s_o_1 + h * s_o_2 + offs, state, mask=mask)


def _run_scan_fwd(h_init, gates, inputs):
    """Launch the forward scan Triton kernel.

    Args:
        h_init: (B, H, DD) contiguous initial state
        gates:  (B, T, H)  contiguous scalar gates
        inputs: (B, T, H, DD) contiguous per-timestep inputs

    Returns:
        h_all: (B, T, H, DD) all timestep states
    """
    B, T, H, DD = inputs.shape
    h_all = torch.empty_like(inputs)

    BLOCK = min(1024, triton.next_power_of_2(DD))
    grid = (B * H * triton.cdiv(DD, BLOCK),)

    _scan_fwd_kernel[grid](
        h_init, gates, inputs, h_all,
        T,
        h_init.stride(0), h_init.stride(1),
        gates.stride(0), gates.stride(1), gates.stride(2),
        inputs.stride(0), inputs.stride(1), inputs.stride(2),
        h_all.stride(0), h_all.stride(1), h_all.stride(2),
        H=H, DD=DD, BLOCK=BLOCK,
    )
    return h_all


class _LinearScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_init, gates, inputs):
        """
        Args:
            h_init: (B, H, D, D) initial state
            gates:  (B, T, H) scalar gates per head per timestep
            inputs: (B, T, H, D, D) per-timestep inputs

        Returns:
            h_all:   (B, T, H, D, D) all intermediate states
            h_final: (B, H, D, D) final state (= h_all[:, -1])
        """
        B, T, H = gates.shape
        D1, D2 = h_init.shape[-2], h_init.shape[-1]
        DD = D1 * D2

        h_init_flat = h_init.reshape(B, H, DD).contiguous()
        inputs_flat = inputs.reshape(B, T, H, DD).contiguous()
        gates_c = gates.contiguous()

        h_all_flat = _run_scan_fwd(h_init_flat, gates_c, inputs_flat)

        h_all = h_all_flat.reshape(B, T, H, D1, D2)
        h_final = h_all[:, -1].contiguous()

        ctx.save_for_backward(h_init_flat, h_all_flat, gates_c)
        ctx.shape_info = (B, T, H, D1, D2, DD)
        return h_all, h_final

    @staticmethod
    def backward(ctx, grad_h_all, grad_h_final):
        h_init_flat, h_all_flat, gates = ctx.saved_tensors
        B, T, H, D1, D2, DD = ctx.shape_info

        grad_h_all_flat = grad_h_all.reshape(B, T, H, DD).contiguous()
        grad_h_final_flat = grad_h_final.reshape(B, H, DD).contiguous()

        # Combine direct and final-state gradients
        effective_grad = grad_h_all_flat.clone()
        effective_grad[:, -1] += grad_h_final_flat

        # === Reverse scan for accumulated gradients ===
        # dh_acc[t] = effective_grad[t] + gate[t+1] * dh_acc[t+1]   (t from T-1 to 0)
        #
        # Rewrite as forward scan on time-reversed sequence:
        #   reversed_acc[t'] = rev_gate[t'] * reversed_acc[t'-1] + reversed_grad[t']
        # where rev_gate[0]=0, rev_gate[t'] = gate[T-t'] for t'>=1
        reversed_grad = effective_grad.flip(1).contiguous()

        rev_gates = torch.zeros_like(gates)
        if T > 1:
            rev_gates[:, 1:] = gates[:, 1:].flip(1)
        rev_gates = rev_gates.contiguous()

        zero_init = torch.zeros(B, H, DD, device=h_init_flat.device, dtype=h_init_flat.dtype)
        dh_acc_reversed = _run_scan_fwd(zero_init, rev_gates, reversed_grad)
        dh_acc = dh_acc_reversed.flip(1).contiguous()  # (B, T, H, DD)

        # === Compute parameter gradients ===
        # grad_input[t] = dh_acc[t]
        grad_inputs = dh_acc.reshape(B, T, H, D1, D2)

        # grad_gate[t] = sum_{d} (dh_acc[t,d] * h_{t-1,d})  (scalar per b,t,h)
        h_prev = torch.cat([h_init_flat.unsqueeze(1), h_all_flat[:, :-1]], dim=1)  # (B, T, H, DD)
        grad_gates = (dh_acc * h_prev).sum(dim=-1)  # (B, T, H)

        # grad_h_init = gate[0] * dh_acc[0]  (element-wise broadcast)
        grad_h_init = (gates[:, 0].unsqueeze(-1) * dh_acc[:, 0]).reshape(B, H, D1, D2)

        return grad_h_init, grad_gates, grad_inputs


# =============================================================================
# Fused Polar Express kernel
# =============================================================================

# Polar Express coefficients (same as atlas.py)
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

_COEFFS_TENSOR_CACHE = {}


def _get_coeffs_tensor(steps, device):
    """Get or create the coefficients tensor for the PE kernel."""
    key = (steps, device)
    if key not in _COEFFS_TENSOR_CACHE:
        flat = [v for abc in _POLAR_EXPRESS_COEFFS[:steps] for v in abc]
        _COEFFS_TENSOR_CACHE[key] = torch.tensor(flat, device=device, dtype=torch.float32)
    return _COEFFS_TENSOR_CACHE[key]


@triton.jit
def _polar_express_fwd_kernel(
    X_ptr, OUT_ptr, coeffs_ptr,
    batch_stride,
    NS_STEPS: tl.constexpr,
    D: tl.constexpr,
    PAD_D: tl.constexpr,
):
    """Fused Polar Express forward: one program per D×D matrix.

    All 5 Newton-Schulz iterations are computed in a single kernel,
    keeping the matrix in registers/SRAM across iterations.
    D is the actual matrix size; PAD_D is the padded power-of-2 for tl.arange.
    """
    bid = tl.program_id(0)
    base = X_ptr + bid * batch_stride
    out_base = OUT_ptr + bid * batch_stride

    rows = tl.arange(0, PAD_D)
    cols = tl.arange(0, PAD_D)
    mask = (rows[:, None] < D) & (cols[None, :] < D)

    # Load D×D matrix (padded to PAD_D×PAD_D with zeros)
    X = tl.load(base + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    # Frobenius normalize
    norm = tl.sqrt(tl.sum(X * X))
    X = X / (norm * 1.01 + 1e-6)

    # Newton-Schulz iterations
    for step in range(NS_STEPS):
        a = tl.load(coeffs_ptr + step * 3).to(tl.float32)
        b = tl.load(coeffs_ptr + step * 3 + 1).to(tl.float32)
        c = tl.load(coeffs_ptr + step * 3 + 2).to(tl.float32)

        A = tl.dot(X, tl.trans(X))        # X @ X.T
        AA = tl.dot(A, A)                 # A @ A
        B = b * A + c * AA                # b*A + c*A²
        X = a * X + tl.dot(B, X)          # a*X + B@X

    # Store result (only the valid D×D region)
    tl.store(out_base + rows[:, None] * D + cols[None, :], X, mask=mask)


# Track whether the fused kernel is available on this GPU
_PE_KERNEL_AVAILABLE = None


def fused_polar_express(X, steps=5):
    """Fused Polar Express orthogonalization via Triton.

    Falls back to the standard PyTorch implementation if the Triton kernel
    can't fit the matrix in shared memory (GPU-dependent) or if the matrix
    is non-square (deep MLP memory with expansion > 1).

    Args:
        X: (..., D1, D2) batch of matrices (square or rectangular)
        steps: number of Newton-Schulz iterations

    Returns:
        Approximate orthogonal polar factor of each matrix.
    """
    global _PE_KERNEL_AVAILABLE
    D1, D2 = X.shape[-2], X.shape[-1]

    # Non-square matrices: fall back to PyTorch (NS iteration works for rectangular)
    if D1 != D2:
        return _polar_express_pytorch(X, steps)

    D = D1
    PAD_D = triton.next_power_of_2(D)

    # Check if kernel is available (cache result per D)
    if _PE_KERNEL_AVAILABLE is None:
        _PE_KERNEL_AVAILABLE = {}
    if D not in _PE_KERNEL_AVAILABLE:
        try:
            # Test with a small batch
            test = torch.randn(1, D, D, device=X.device, dtype=X.dtype)
            out = torch.empty_like(test)
            coeffs = _get_coeffs_tensor(steps, X.device)
            _polar_express_fwd_kernel[(1,)](
                test, out, coeffs, D * D, NS_STEPS=steps, D=D, PAD_D=PAD_D,
            )
            _PE_KERNEL_AVAILABLE[D] = True
        except Exception:
            _PE_KERNEL_AVAILABLE[D] = False

    if not _PE_KERNEL_AVAILABLE.get(D, False):
        return _polar_express_pytorch(X, steps)

    shape = X.shape
    X_flat = X.reshape(-1, D, D).contiguous()
    N = X_flat.shape[0]
    OUT = torch.empty_like(X_flat)
    coeffs = _get_coeffs_tensor(steps, X.device)

    _polar_express_fwd_kernel[(N,)](
        X_flat, OUT, coeffs, D * D, NS_STEPS=steps, D=D, PAD_D=PAD_D,
    )
    return OUT.reshape(shape)


def _polar_express_pytorch(X, steps=5):
    """Standard PyTorch polar express (fallback)."""
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in _POLAR_EXPRESS_COEFFS[:steps]:
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


class _PolarExpressSTE(torch.autograd.Function):
    """Polar Express with Straight-Through Estimator for backward.

    Forward: full Newton-Schulz orthogonalization via fused_polar_express.
    Backward: pass gradients through unchanged (identity Jacobian approximation).

    Justified because PE is an internal optimizer step finding the nearest
    orthogonal matrix. Near-orthogonal matrices have Jacobian ≈ identity.
    Eliminates 89%+ of backward FLOPs (30 cuBLAS matmuls per PE call)."""

    @staticmethod
    def forward(ctx, X, steps):
        return fused_polar_express(X, steps)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def polar_express_ste(X, steps=5):
    """Polar Express with straight-through estimator for backward pass."""
    return _PolarExpressSTE.apply(X, steps)


def fused_linear_scan(h_init, gates, inputs):
    """Fused linear scan: h_t = gate_t * h_{t-1} + input_t.

    Replaces a sequential Python for-loop with a single Triton kernel launch.
    Supports autograd for both forward and backward passes.

    Args:
        h_init: (B, H, D, D) initial state matrix per head
        gates:  (B, T, H) scalar gate per head per timestep
        inputs: (B, T, H, D, D) per-timestep input matrices

    Returns:
        h_all:   (B, T, H, D, D) all intermediate states
        h_final: (B, H, D, D) final state
    """
    return _LinearScan.apply(h_init, gates, inputs)
