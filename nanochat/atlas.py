"""
Atlas: Learning to Optimally Memorize the Context at Test Time
arXiv: 2505.23735

A recurrent architecture that replaces attention with a deep memory module
updated by the Omega rule with Muon-style optimizer (Newton-Schulz / Polar Express
orthogonalization).

Key features:
- Deep MLP memory per head (2-layer with residual), or linear matrix memory (configurable)
- Omega rule: sliding window gradient aggregation with per-token context gates (γ_i)
- Polynomial feature mapping on keys/queries (learnable coefficients ≈ Taylor of exp)
- Polar Express orthogonalization for locally optimal memory management
- Input-dependent forgetting (α), learning rate (η), momentum (θ), and context (γ) gates
- Short causal convolution on Q, K, V projections
- Chunk-parallel computation for efficient training with gradient checkpointing
- No positional encoding needed (recurrent by nature)

Notable differences from nanochat GPT (gpt.py):
- Recurrent memory replaces attention — O(n) instead of O(n²) in sequence length
- GELU activation in MLP (paper convention) instead of relu²
- No rotary embeddings, sliding windows, or KV cache
- Memory state carries across sequence positions (and optionally across calls for inference)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.atlas_kernels import fused_linear_scan, fused_polar_express, polar_express_ste
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class AtlasConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    chunk_size: int = 64    # tokens per chunk for parallel memory computation
    conv_kernel: int = 4    # short causal convolution kernel size
    ns_steps: int = 5       # Polar Express orthogonalization iterations
    omega_window: int = 16  # Omega rule sliding window size (1 = online/Delta rule)
    poly_degree: int = 3    # polynomial feature mapping degree (0 = disabled)
    deep_memory: bool = True   # deep MLP memory vs linear matrix memory
    memory_expand: int = 1     # MLP expansion factor for deep memory (1 = D×D weights)
    pe_ste: bool = False       # Polar Express straight-through estimator (skip PE backward)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Master weights stay fp32 for optimizer precision, matmuls run in activation dtype."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# Polar Express coefficients (from nanochat/optim.py, arxiv 2505.16932)
# Per-step (a, b, c) tuples for the orthogonalization iteration
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(X, steps=5):
    """Batched Polar Express orthogonalization (improved Newton-Schulz).
    X: (..., d, d) batch of square matrices.
    Returns approximate orthogonal polar factor of each matrix.

    This is the same algorithm used in nanochat's Muon optimizer for weight
    orthogonalization, here repurposed as Atlas's internal memory optimizer."""
    # Frobenius-normalize each matrix
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    for a, b, c in _POLAR_EXPRESS_COEFFS[:steps]:
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _gelu_derivative(x):
    """Exact derivative of GELU(x) = x * Φ(x) where Φ is the standard normal CDF."""
    cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
    pdf = torch.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


def _omega_aggregate(u, gamma, omega_window):
    """Sliding window aggregation with per-position context gates (Omega rule).

    For each position t, computes: Σ_{i=max(0,t-w+1)}^{t} γ_i * u_i
    Uses cumsum for O(n) computation instead of O(n*w).

    Args:
        u: (B, cs, H, ...) per-position gradient values
        gamma: (B, cs, H, 1) per-position context gates
        omega_window: sliding window size
    """
    cs = u.shape[1]
    g = gamma
    while g.ndim < u.ndim:
        g = g.unsqueeze(-1)
    weighted = g * u
    cum = torch.cumsum(weighted, dim=1)
    if omega_window >= cs:
        return cum
    result = cum.clone()
    result[:, omega_window:] = cum[:, omega_window:] - cum[:, :-omega_window]
    return result


class ShortConv(nn.Module):
    """Causal depthwise 1D convolution (per Titans / Based convention)."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (B, T, D) -> conv over time dim, independent per channel
        x = x.transpose(1, 2)                          # (B, D, T)
        x = F.pad(x, (self.kernel_size - 1, 0))        # causal left-padding
        x = F.conv1d(x, self.weight.to(x.dtype), self.bias.to(x.dtype), groups=x.size(1))
        return x.transpose(1, 2)                        # (B, T, D)


class AtlasMemoryLayer(nn.Module):
    """Multi-head linear memory layer with Omega rule + Polar Express update.

    Each head maintains a memory matrix M in R^{d x d} and a momentum matrix S.
    Per token, the update is:
      1. Gradient:     u_t = 2 (M @ k_t - v_t) @ k_t^T
      2. Momentum:     S_t = theta_t * S_{t-1} - eta_t * u_t
      3. Orthogonalize: S'_t = PolarExpress(S_t)
      4. Memory:       M_t = alpha_t * M_{t-1} + S'_t
      5. Output:       y_t = M_t @ q_t

    Computation is chunked: within each chunk, gradients u_t are computed in
    parallel w.r.t. the frozen memory state at the chunk boundary. The momentum
    and memory scans are sequential within each chunk.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = config.chunk_size
        self.ns_steps = config.ns_steps
        self.omega_window = config.omega_window
        self.poly_degree = config.poly_degree
        self.deep_memory = config.deep_memory
        self.memory_expand = config.memory_expand
        self.expand_dim = self.memory_expand * self.head_dim if self.deep_memory else self.head_dim
        self.pe_ste = config.pe_ste
        assert config.n_embd % config.n_head == 0
        assert config.omega_window <= config.chunk_size, \
            f"omega_window ({config.omega_window}) must be <= chunk_size ({config.chunk_size})"

        # Input projections
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

        # Short causal convolutions on Q, K, V
        self.conv_q = ShortConv(config.n_embd, config.conv_kernel)
        self.conv_k = ShortConv(config.n_embd, config.conv_kernel)
        self.conv_v = ShortConv(config.n_embd, config.conv_kernel)

        # Input-dependent gates (scalar per head):
        #   alpha: forgetting factor, eta: learning rate, theta: momentum coefficient
        self.gate_alpha = Linear(config.n_embd, config.n_head, bias=False)
        self.gate_eta = Linear(config.n_embd, config.n_head, bias=False)
        self.gate_theta = Linear(config.n_embd, config.n_head, bias=False)

        # Omega rule: per-token context gates within the sliding window
        if self.omega_window > 1:
            self.gate_gamma = Linear(config.n_embd, config.n_head, bias=False)

        # Polynomial feature mapping: learnable coefficients initialized at 1/i!
        if self.poly_degree > 0:
            coeffs = [1.0 / math.factorial(i) for i in range(1, self.poly_degree + 1)]
            self.poly_coeffs = nn.Parameter(torch.tensor(coeffs))

    def _poly_features(self, x):
        """Element-wise polynomial feature mapping: φ(x) = Σ_{i=1}^{p} a_i * x^i.
        Learnable coefficients approximate the Taylor expansion of exp(q^T k)."""
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, self.poly_degree):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    @staticmethod
    def _process_chunk(M, S, q_c, k_c, v_c, a_c, e_c, t_c, g_c, ns_steps, omega_window,
                       use_pe_ste):
        """Process a single chunk with linear memory: gradient, momentum scan, polar express, memory scan.
        Extracted as a static method so it can be wrapped with torch.utils.checkpoint."""
        _pe = polar_express_ste if use_pe_ste else fused_polar_express

        # --- Parallel: compute per-position gradients w.r.t. frozen memory M ---
        pred = torch.einsum('bhvk,bchk->bchv', M, k_c)
        err = pred - v_c
        u = 2.0 * torch.einsum('bchv,bchk->bchvk', err, k_c)   # outer product

        # --- Omega rule: sliding window aggregation with context gates ---
        if omega_window > 1 and g_c is not None:
            u = _omega_aggregate(u, g_c, omega_window)

        # --- Fused momentum scan ---
        # S_t = theta_t * S_{t-1} - eta_t * u_t
        theta = t_c.squeeze(-1)                          # (B, cs, H)
        momentum_input = -(e_c.unsqueeze(-1) * u)        # (B, cs, H, D, D)
        chunk_S, S = fused_linear_scan(S, theta, momentum_input)

        # --- Polar Express orthogonalization ---
        chunk_S_orth = _pe(chunk_S, ns_steps)

        # --- Fused memory scan ---
        # M_t = alpha_t * M_{t-1} + PolarExpress(S_t)
        alpha = a_c.squeeze(-1)                           # (B, cs, H)
        M_all, M = fused_linear_scan(M, alpha, chunk_S_orth)

        # --- Parallel: output y = M_t @ q_t for all timesteps ---
        y_c = torch.einsum('bchvk,bchk->bchv', M_all, q_c)

        return y_c, M, S

    @staticmethod
    def _process_chunk_deep(W1, W2, S_W1, S_W2, q_c, k_c, v_c, a_c, e_c, t_c, g_c,
                            ns_steps, omega_window, use_pe_ste):
        """Process a single chunk with deep MLP memory.
        Memory is a 2-layer MLP: M(x) = x + W1 @ GELU(W2 @ x) with residual connection.
        State consists of (W1, W2) weights and (S_W1, S_W2) momentum matrices."""
        _pe = polar_express_ste if use_pe_ste else fused_polar_express

        # --- Forward through frozen MLP memory ---
        h = torch.einsum('bhed,bchd->bche', W2, k_c)      # (B, cs, H, E)
        act = F.gelu(h)                                      # (B, cs, H, E)
        y_pred = k_c + torch.einsum('bhde,bche->bchd', W1, act)  # (B, cs, H, D)
        err = y_pred - v_c                                   # (B, cs, H, D)

        # --- Gradients w.r.t. W1: ∂loss/∂W1 = 2 * err ⊗ GELU(W2 @ k) ---
        u_W1 = 2.0 * torch.einsum('bchd,bche->bchde', err, act)   # (B, cs, H, D, E)

        # --- Gradients w.r.t. W2: chain rule through GELU and W1 ---
        gelu_prime = _gelu_derivative(h)                           # (B, cs, H, E)
        w1t_err = torch.einsum('bhde,bchd->bche', W1, err)        # (B, cs, H, E)
        chain = w1t_err * gelu_prime                               # (B, cs, H, E)
        u_W2 = 2.0 * torch.einsum('bche,bchd->bched', chain, k_c) # (B, cs, H, E, D)

        # --- Omega rule: sliding window aggregation ---
        if omega_window > 1 and g_c is not None:
            u_W1 = _omega_aggregate(u_W1, g_c, omega_window)
            u_W2 = _omega_aggregate(u_W2, g_c, omega_window)

        theta = t_c.squeeze(-1)                                    # (B, cs, H)
        alpha = a_c.squeeze(-1)                                    # (B, cs, H)

        # --- Momentum + Polar Express + Memory scan for W1 ---
        mom_W1 = -(e_c.unsqueeze(-1) * u_W1)                      # (B, cs, H, D, E)
        chunk_S_W1, S_W1 = fused_linear_scan(S_W1, theta, mom_W1)
        chunk_S_W1_orth = _pe(chunk_S_W1, ns_steps)
        W1_all, W1 = fused_linear_scan(W1, alpha, chunk_S_W1_orth)

        # --- Momentum + Polar Express + Memory scan for W2 ---
        mom_W2 = -(e_c.unsqueeze(-1) * u_W2)                      # (B, cs, H, E, D)
        chunk_S_W2, S_W2 = fused_linear_scan(S_W2, theta, mom_W2)
        chunk_S_W2_orth = _pe(chunk_S_W2, ns_steps)
        W2_all, W2 = fused_linear_scan(W2, alpha, chunk_S_W2_orth)

        # --- Output: y_t = M_t(q_t) = q_t + W1_t @ GELU(W2_t @ q_t) ---
        h_q = torch.einsum('bched,bchd->bche', W2_all, q_c)      # (B, cs, H, E)
        g_q = F.gelu(h_q)                                          # (B, cs, H, E)
        y_c = q_c + torch.einsum('bchde,bche->bchd', W1_all, g_q) # (B, cs, H, D)

        return y_c, W1, W2, S_W1, S_W2

    def forward(self, x, memory_state=None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        E = self.expand_dim
        cs = self.chunk_size

        # Project and apply short causal convolution, then reshape to multi-head
        q = self.conv_q(self.c_q(x)).view(B, T, H, D)
        k = self.conv_k(self.c_k(x)).view(B, T, H, D)
        v = self.conv_v(self.c_v(x)).view(B, T, H, D)

        # Normalize queries and keys for stable memory operations
        q, k = norm(q), norm(k)

        # Polynomial feature mapping on keys and queries
        if self.poly_degree > 0:
            q = self._poly_features(q)
            k = self._poly_features(k)

        # Compute input-dependent gates via sigmoid -> (0, 1) range
        alpha = torch.sigmoid(self.gate_alpha(x)).view(B, T, H, 1)
        eta   = torch.sigmoid(self.gate_eta(x)).view(B, T, H, 1)
        theta = torch.sigmoid(self.gate_theta(x)).view(B, T, H, 1)

        # Omega rule context gates
        gamma = None
        if self.omega_window > 1:
            gamma = torch.sigmoid(self.gate_gamma(x)).view(B, T, H, 1)

        # Initialize or unpack memory state
        if memory_state is None:
            if self.deep_memory:
                W1 = torch.zeros(B, H, D, E, device=x.device, dtype=x.dtype)
                # W2 initialized to [I; 0] so GELU(W2 @ k) ≠ 0, enabling learning from step 1
                W2 = torch.zeros(B, H, E, D, device=x.device, dtype=x.dtype)
                eye = torch.eye(min(E, D), device=x.device, dtype=x.dtype)
                W2[:, :, :min(E, D), :min(E, D)] = eye
                S_W1 = torch.zeros(B, H, D, E, device=x.device, dtype=x.dtype)
                S_W2 = torch.zeros(B, H, E, D, device=x.device, dtype=x.dtype)
            else:
                M = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
                S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        else:
            if self.deep_memory:
                W1, W2, S_W1, S_W2 = memory_state
            else:
                M, S = memory_state

        # Pad sequence to a multiple of chunk_size
        T_orig = T
        if T % cs != 0:
            pad = cs - T % cs
            q     = F.pad(q,     (0, 0, 0, 0, 0, pad))
            k     = F.pad(k,     (0, 0, 0, 0, 0, pad))
            v     = F.pad(v,     (0, 0, 0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, 0, 0, pad), value=1.0)   # carry memory unchanged
            eta   = F.pad(eta,   (0, 0, 0, 0, 0, pad), value=0.0)   # no gradient update
            theta = F.pad(theta, (0, 0, 0, 0, 0, pad), value=0.0)   # kill momentum
            if gamma is not None:
                gamma = F.pad(gamma, (0, 0, 0, 0, 0, pad), value=0.0)
            T = q.shape[1]

        n_chunks = T // cs
        outputs = []

        for ci in range(n_chunks):
            s, e = ci * cs, (ci + 1) * cs
            q_c, k_c, v_c = q[:, s:e], k[:, s:e], v[:, s:e]
            a_c, e_c, t_c = alpha[:, s:e], eta[:, s:e], theta[:, s:e]
            g_c = gamma[:, s:e] if gamma is not None else None

            # Gradient checkpointing: only chunk-boundary states are stored;
            # all intra-chunk intermediates are recomputed during backward.
            if self.deep_memory:
                y_c, W1, W2, S_W1, S_W2 = checkpoint(
                    self._process_chunk_deep,
                    W1, W2, S_W1, S_W2, q_c, k_c, v_c, a_c, e_c, t_c, g_c,
                    self.ns_steps, self.omega_window, self.pe_ste,
                    use_reentrant=False,
                )
            else:
                y_c, M, S = checkpoint(
                    self._process_chunk, M, S, q_c, k_c, v_c, a_c, e_c, t_c, g_c,
                    self.ns_steps, self.omega_window, self.pe_ste,
                    use_reentrant=False,
                )

            outputs.append(y_c)

        # Concatenate chunk outputs, trim padding, project back to residual stream
        y = torch.cat(outputs, dim=1)[:, :T_orig]
        y = y.contiguous().view(B, T_orig, -1)
        y = self.c_proj(y)

        if self.deep_memory:
            return y, (W1, W2, S_W1, S_W2)
        return y, (M, S)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory = AtlasMemoryLayer(config)
        self.mlp = MLP(config)

    def forward(self, x, memory_state=None):
        mem_out, new_state = self.memory(norm(x), memory_state)
        x = x + mem_out
        x = x + self.mlp(norm(x))
        return x, new_state


class MemoryState:
    """Holds per-layer memory and momentum matrices for inference.
    During training, states are created fresh per forward pass and discarded.
    During inference, this object persists across generate() calls."""
    def __init__(self, n_layers, batch_size, n_head, head_dim, device, dtype,
                 deep_memory=False, expand_dim=None):
        self.n_layers = n_layers
        self.deep_memory = deep_memory
        if deep_memory:
            E = expand_dim or head_dim
            self.W1 = [torch.zeros(batch_size, n_head, head_dim, E, device=device, dtype=dtype)
                       for _ in range(n_layers)]
            self.W2 = []
            for _ in range(n_layers):
                w2 = torch.zeros(batch_size, n_head, E, head_dim, device=device, dtype=dtype)
                eye = torch.eye(min(E, head_dim), device=device, dtype=dtype)
                w2[:, :, :min(E, head_dim), :min(E, head_dim)] = eye
                self.W2.append(w2)
            self.S_W1 = [torch.zeros(batch_size, n_head, head_dim, E, device=device, dtype=dtype)
                         for _ in range(n_layers)]
            self.S_W2 = [torch.zeros(batch_size, n_head, E, head_dim, device=device, dtype=dtype)
                         for _ in range(n_layers)]
        else:
            self.M = [torch.zeros(batch_size, n_head, head_dim, head_dim, device=device, dtype=dtype)
                       for _ in range(n_layers)]
            self.S = [torch.zeros(batch_size, n_head, head_dim, head_dim, device=device, dtype=dtype)
                       for _ in range(n_layers)]

    def get_layer_state(self, layer_idx):
        if self.deep_memory:
            return (self.W1[layer_idx], self.W2[layer_idx],
                    self.S_W1[layer_idx], self.S_W2[layer_idx])
        return (self.M[layer_idx], self.S[layer_idx])

    def set_layer_state(self, layer_idx, *state):
        if self.deep_memory:
            self.W1[layer_idx], self.W2[layer_idx] = state[0], state[1]
            self.S_W1[layer_idx], self.S_W2[layer_idx] = state[2], state[3]
        else:
            self.M[layer_idx] = state[0]
            self.S[layer_idx] = state[1]


class Atlas(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """NOTE: designed to run in meta device context — shapes/dtypes only, no data.
        Actual parameter initialization happens in init_weights()."""
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all parameters. Called after model is moved to a real device."""
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init for projections, zero for output projections
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) * 1/sqrt(n_embd): uniform gives same std as normal

        for block in self.transformer.h:
            mem = block.memory
            # Q, K, V input projections
            torch.nn.init.uniform_(mem.c_q.weight, -s, s)
            torch.nn.init.uniform_(mem.c_k.weight, -s, s)
            torch.nn.init.uniform_(mem.c_v.weight, -s, s)
            torch.nn.init.zeros_(mem.c_proj.weight)     # output projection starts at zero
            # Short convolutions: small random init
            torch.nn.init.normal_(mem.conv_q.weight, std=0.02)
            torch.nn.init.normal_(mem.conv_k.weight, std=0.02)
            torch.nn.init.normal_(mem.conv_v.weight, std=0.02)
            # Gates: small init so sigmoid(x) ≈ 0.5 (neutral starting point)
            torch.nn.init.normal_(mem.gate_alpha.weight, std=0.01)
            torch.nn.init.normal_(mem.gate_eta.weight, std=0.01)
            torch.nn.init.normal_(mem.gate_theta.weight, std=0.01)
            if hasattr(mem, 'gate_gamma'):
                torch.nn.init.normal_(mem.gate_gamma.weight, std=0.01)
            # poly_coeffs: already initialized in __init__ to 1/i!
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Cast embeddings to compute dtype for memory savings (same as gpt.py)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Estimate FLOPs per token (forward + backward).
        Atlas replaces O(n²) attention FLOPs with O(n * H * D²) memory operations."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()  # embedding is a lookup, not a matmul
        H = self.config.n_head
        D = self.config.n_embd // H
        E = self.config.memory_expand * D if self.config.deep_memory else D
        if self.config.deep_memory:
            # Deep MLP memory: W1(D,E) and W2(E,D), dual scans, dual polar express
            elementwise_flops = H * (D * E + E * D) * 5  # gradient + scans + output (×2 matrices)
            ns_flops = 2 * 3 * self.config.ns_steps * 2 * H * max(D, E)**3  # PE on both
        else:
            elementwise_flops = H * D * D * 5
            ns_flops = 3 * self.config.ns_steps * 2 * H * D * D * D
        memory_flops_per_token = elementwise_flops + ns_flops
        total_memory_flops = memory_flops_per_token * self.config.n_layer
        return 6 * (nparams - nparams_exclude) + total_memory_flops

    def num_scaling_params(self):
        """Parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Separate 2D weight matrices (Muon) from smaller/non-2D params (AdamW)
        matrix_params = []
        small_params = []   # conv weights (3D), conv biases (1D)
        for block in self.transformer.h:
            for name, p in block.named_parameters():
                if p.ndim == 2:
                    matrix_params.append(p)
                else:
                    small_params.append(p)

        all_count = len(embedding_params) + len(lm_head_params) + len(matrix_params) + len(small_params)
        assert all_count == len(list(self.parameters())), f"Parameter grouping mismatch: {all_count} vs {len(list(self.parameters()))}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR for AdamW params ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=small_params, lr=scalar_lr * 0.1,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups: stack 2D matrices by shape for efficient batch orthogonalization
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, memory_state=None, loss_reduction='mean'):
        B, T = idx.size()

        # Embed tokens and normalize
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Forward through all blocks, threading memory state through each layer
        for i, block in enumerate(self.transformer.h):
            layer_state = memory_state.get_layer_state(i) if memory_state is not None else None
            x, new_state = block(x, layer_state)
            if memory_state is not None:
                memory_state.set_layer_state(i, *new_state)

        x = norm(x)

        # Compute logits with soft capping (same as gpt.py)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive streaming inference with persistent memory state."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Initialize persistent memory state
        H = self.config.n_head
        D = self.config.n_embd // H
        E = self.config.memory_expand * D if self.config.deep_memory else D
        state = MemoryState(self.config.n_layer, 1, H, D, device, COMPUTE_DTYPE,
                            deep_memory=self.config.deep_memory, expand_dim=E)

        # Prefill: process the full prompt, evolving memory over all positions
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.forward(ids, memory_state=state)

        for _ in range(max_tokens):
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            token = next_ids.item()
            yield token
            # Process new token with carried-forward memory state
            logits = self.forward(next_ids, memory_state=state)
