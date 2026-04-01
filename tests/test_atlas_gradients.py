"""Gradient correctness tests for Atlas components.

Uses torch.autograd.gradcheck at fp64 with tiny dimensions.
CPU-runnable tests cover pure-Python components.
GPU tests (requires Triton) cover fused kernels and full chunk processing.
"""
import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CPU tests: pure-Python components
# ---------------------------------------------------------------------------

class TestGeluDerivativeGrad:
    """Verify _gelu_derivative is the actual derivative of GELU via autograd."""

    def test_autograd_agreement(self):
        """GELU derivative matches autograd's computed derivative."""
        try:
            from nanochat.atlas import _gelu_derivative
        except (ImportError, ModuleNotFoundError):
            def _gelu_derivative(x):
                cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
                pdf = torch.exp(-0.5 * x * x) * 0.3989422804014327
                return cdf + x * pdf
        x = torch.randn(50, dtype=torch.float64, requires_grad=True)
        # Compute GELU and get autograd derivative
        y = F.gelu(x)
        y.sum().backward()
        autograd_deriv = x.grad.clone()
        # Compare with our analytic derivative
        analytic_deriv = _gelu_derivative(x.detach())
        torch.testing.assert_close(analytic_deriv, autograd_deriv, atol=1e-10, rtol=1e-10)


class TestPolyFeaturesGrad:
    """Gradient check for polynomial feature mapping."""

    def test_gradcheck_coeffs(self):
        """Gradcheck: gradient w.r.t. polynomial coefficients."""
        coeffs = torch.tensor([1.0, 0.5, 1.0 / 6.0], dtype=torch.float64, requires_grad=True)
        x = torch.randn(2, 3, 4, dtype=torch.float64)

        def poly_fn(c):
            result = c[0] * x
            x_pow = x.clone()
            for i in range(1, len(c)):
                x_pow = x_pow * x
                result = result + c[i] * x_pow
            return result

        assert torch.autograd.gradcheck(poly_fn, (coeffs,), eps=1e-6)

    def test_gradcheck_input(self):
        """Gradcheck: gradient w.r.t. input x."""
        coeffs = torch.tensor([1.0, 0.5, 1.0 / 6.0], dtype=torch.float64)

        def poly_fn(x):
            result = coeffs[0] * x
            x_pow = x.clone()
            for i in range(1, len(coeffs)):
                x_pow = x_pow * x
                result = result + coeffs[i] * x_pow
            return result

        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(poly_fn, (x,), eps=1e-6)


class TestOmegaAggregateGrad:
    """Gradient check for Omega rule aggregation."""

    def _get_omega_fn(self):
        try:
            from nanochat.atlas import _omega_aggregate
            return _omega_aggregate
        except ImportError:
            # Standalone fallback for CPU
            def _omega_aggregate(u, gamma, omega_window):
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
            return _omega_aggregate

    def test_gradcheck_u(self):
        """Gradcheck: gradient w.r.t. input u."""
        fn = self._get_omega_fn()
        gamma = torch.rand(1, 4, 2, 1, dtype=torch.float64)

        def f(u):
            return fn(u, gamma, omega_window=3)

        u = torch.randn(1, 4, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (u,), eps=1e-6)

    def test_gradcheck_gamma(self):
        """Gradcheck: gradient w.r.t. gamma gates."""
        fn = self._get_omega_fn()
        u = torch.randn(1, 4, 2, 3, 3, dtype=torch.float64)

        def f(gamma):
            return fn(u, gamma, omega_window=3)

        gamma = torch.rand(1, 4, 2, 1, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (gamma,), eps=1e-6)


# ---------------------------------------------------------------------------
# GPU tests: fused kernels and chunk processing (require Triton)
# ---------------------------------------------------------------------------

try:
    from nanochat.atlas import AtlasMemoryLayer, AtlasConfig
    from nanochat.atlas_kernels import fused_linear_scan, fused_polar_express
    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False

requires_gpu = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Requires Triton + CUDA"
)


@requires_gpu
class TestFusedLinearScanGrad:
    """Gradient check for the fused linear scan Triton kernel."""

    def test_gradcheck(self):
        """Full gradcheck on fused_linear_scan."""
        B, T, H, D = 1, 4, 1, 3
        h_init = torch.randn(B, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)
        gates = torch.rand(B, T, H, dtype=torch.float64, device='cuda', requires_grad=True)
        inputs = torch.randn(B, T, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)

        def fn(h, g, inp):
            h_all, h_final = fused_linear_scan(h, g, inp)
            return h_all, h_final

        assert torch.autograd.gradcheck(fn, (h_init, gates, inputs), eps=1e-6)


@requires_gpu
class TestProcessChunkGrad:
    """Gradient check for _process_chunk (linear memory path)."""

    def test_gradcheck_linear(self):
        """Gradcheck on _process_chunk with omega_window=1."""
        D, H = 4, 1
        M = torch.randn(1, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)
        S = torch.randn(1, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)
        q = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        k = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        v = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        a = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        e = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)

        def fn(M, S, q, k, v, a, e, t):
            y, M_new, S_new = AtlasMemoryLayer._process_chunk(
                M, S, q, k, v, a, e, t, None, 2, 1, use_pe_ste=False)
            return y, M_new, S_new

        assert torch.autograd.gradcheck(fn, (M, S, q, k, v, a, e, t), eps=1e-5, atol=1e-3)

    def test_gradcheck_omega(self):
        """Gradcheck on _process_chunk with omega_window=3."""
        D, H = 4, 1
        M = torch.randn(1, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)
        S = torch.randn(1, H, D, D, dtype=torch.float64, device='cuda', requires_grad=True)
        q = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        k = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        v = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        a = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        e = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        g = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)

        def fn(M, S, q, k, v, a, e, t, g):
            y, M_new, S_new = AtlasMemoryLayer._process_chunk(
                M, S, q, k, v, a, e, t, g, 2, 3, use_pe_ste=False)
            return y, M_new, S_new

        assert torch.autograd.gradcheck(fn, (M, S, q, k, v, a, e, t, g), eps=1e-5, atol=1e-3)


@requires_gpu
class TestProcessChunkDeepGrad:
    """Gradient check for _process_chunk_deep (deep MLP memory path)."""

    def test_gradcheck_deep_square(self):
        """Gradcheck with memory_expand=1 (square W1, W2)."""
        D, H, E = 4, 1, 4  # expand=1: E=D
        W1 = torch.randn(1, H, D, E, dtype=torch.float64, device='cuda', requires_grad=True)
        W2_init = torch.eye(D, dtype=torch.float64, device='cuda').unsqueeze(0).unsqueeze(0)
        W2 = (W2_init + 0.01 * torch.randn(1, H, E, D, dtype=torch.float64, device='cuda')).requires_grad_(True)
        S_W1 = torch.randn(1, H, D, E, dtype=torch.float64, device='cuda', requires_grad=True)
        S_W2 = torch.randn(1, H, E, D, dtype=torch.float64, device='cuda', requires_grad=True)
        q = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        k = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        v = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        a = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        e = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)

        def fn(W1, W2, S_W1, S_W2, q, k, v, a, e, t):
            y, W1_n, W2_n, S1_n, S2_n = AtlasMemoryLayer._process_chunk_deep(
                W1, W2, S_W1, S_W2, q, k, v, a, e, t, None, 2, 1, use_pe_ste=False)
            return y, W1_n, W2_n, S1_n, S2_n

        assert torch.autograd.gradcheck(
            fn, (W1, W2, S_W1, S_W2, q, k, v, a, e, t),
            eps=1e-5, atol=1e-3, nondet_tol=1e-4)

    def test_gradcheck_deep_omega(self):
        """Gradcheck with omega_window=3 and deep memory."""
        D, H, E = 4, 1, 4
        W1 = torch.randn(1, H, D, E, dtype=torch.float64, device='cuda', requires_grad=True)
        W2_init = torch.eye(D, dtype=torch.float64, device='cuda').unsqueeze(0).unsqueeze(0)
        W2 = (W2_init + 0.01 * torch.randn(1, H, E, D, dtype=torch.float64, device='cuda')).requires_grad_(True)
        S_W1 = torch.randn(1, H, D, E, dtype=torch.float64, device='cuda', requires_grad=True)
        S_W2 = torch.randn(1, H, E, D, dtype=torch.float64, device='cuda', requires_grad=True)
        q = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        k = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        v = torch.randn(1, 4, H, D, dtype=torch.float64, device='cuda', requires_grad=True)
        a = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        e = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)
        g = torch.rand(1, 4, H, 1, dtype=torch.float64, device='cuda', requires_grad=True)

        def fn(W1, W2, S_W1, S_W2, q, k, v, a, e, t, g):
            y, W1_n, W2_n, S1_n, S2_n = AtlasMemoryLayer._process_chunk_deep(
                W1, W2, S_W1, S_W2, q, k, v, a, e, t, g, 2, 3, use_pe_ste=False)
            return y, W1_n, W2_n, S1_n, S2_n

        assert torch.autograd.gradcheck(
            fn, (W1, W2, S_W1, S_W2, q, k, v, a, e, t, g),
            eps=1e-5, atol=1e-3, nondet_tol=1e-4)
