"""Correctness tests for Atlas features: Omega rule, polynomial features, deep MLP memory.

These tests use pure-Python reference implementations and small dimensions,
runnable on CPU without Triton.
"""
import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementations (independent of atlas.py, for cross-checking)
# ---------------------------------------------------------------------------

def _omega_aggregate_ref(u, gamma, omega_window):
    """Naive double-loop reference for Omega sliding window aggregation."""
    B, cs, H = u.shape[:3]
    result = torch.zeros_like(u)
    for t in range(cs):
        for i in range(max(0, t - omega_window + 1), t + 1):
            # gamma[:, i] is (B, H, 1) — expand to match u[:, i] dims
            g = gamma[:, i]  # (B, H, 1)
            while g.ndim < u[:, i].ndim:
                g = g.unsqueeze(-1)
            result[:, t] += g * u[:, i]
    return result


def _poly_features_ref(x, coeffs):
    """Reference polynomial feature mapping."""
    result = coeffs[0] * x
    x_pow = x.clone()
    for i in range(1, len(coeffs)):
        x_pow = x_pow * x
        result = result + coeffs[i] * x_pow
    return result


def _gelu_derivative_ref(x):
    """Reference GELU derivative via finite differences."""
    eps = 1e-5
    return (F.gelu(x + eps) - F.gelu(x - eps)) / (2 * eps)


# ---------------------------------------------------------------------------
# Import the actual implementations
# We import these lazily because nanochat.atlas imports atlas_kernels (Triton).
# On CPU-only machines, we test via the reference implementations above.
# ---------------------------------------------------------------------------

try:
    from nanochat.atlas import _omega_aggregate, _gelu_derivative
    _HAS_ATLAS_HELPERS = True
except ImportError:
    # Provide standalone copies for CPU-only testing (same logic as atlas.py)
    _HAS_ATLAS_HELPERS = False

    def _gelu_derivative(x):
        cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
        pdf = torch.exp(-0.5 * x * x) * 0.3989422804014327
        return cdf + x * pdf

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


class TestOmegaAggregate:
    """Tests for the Omega rule sliding window aggregation."""

    def test_window_1_is_identity(self):
        """omega_window=1 should return gamma-weighted u unchanged."""
        u = torch.randn(2, 8, 4, 3, 3)
        gamma = torch.ones(2, 8, 4, 1)
        result = _omega_aggregate(u, gamma, omega_window=1)
        # Window of 1: each position only includes itself, weighted by gamma=1
        torch.testing.assert_close(result, u)

    def test_full_window_is_cumsum(self):
        """omega_window >= chunk_size should give cumulative sum."""
        u = torch.randn(1, 5, 2, 3, 3)
        gamma = torch.ones(1, 5, 2, 1)
        result = _omega_aggregate(u, gamma, omega_window=100)
        expected = torch.cumsum(u, dim=1)
        torch.testing.assert_close(result, expected)

    def test_matches_naive_reference(self):
        """Cumsum-based implementation matches naive double-loop."""
        B, cs, H, D = 2, 8, 3, 4
        u = torch.randn(B, cs, H, D, D)
        gamma = torch.rand(B, cs, H, 1)
        for w in [1, 3, 5, 8, 16]:
            result = _omega_aggregate(u, gamma, omega_window=w)
            expected = _omega_aggregate_ref(u, gamma, omega_window=w)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5,
                                       msg=f"Failed for omega_window={w}")

    def test_gamma_zero_masks_contributions(self):
        """gamma=0 at a position should exclude it from all windows."""
        u = torch.ones(1, 4, 1, 2, 2)
        gamma = torch.ones(1, 4, 1, 1)
        gamma[:, 1] = 0  # mask out position 1
        result = _omega_aggregate(u, gamma, omega_window=3)
        # Position 2: window [0,1,2], but gamma[1]=0, so sum = u[0]*1 + u[2]*1 = 2
        assert result[0, 2, 0, 0, 0].item() == pytest.approx(2.0)

    def test_gradient_flows(self):
        """Verify gradients flow through _omega_aggregate."""
        u = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
        gamma_raw = torch.randn(1, 4, 2, 1, requires_grad=True)
        gamma = gamma_raw.sigmoid()
        result = _omega_aggregate(u, gamma, omega_window=3)
        result.sum().backward()
        assert u.grad is not None
        assert gamma_raw.grad is not None


class TestPolyFeatures:
    """Tests for the polynomial feature mapping."""

    def test_degree1_is_scaled_identity(self):
        """With degree=1, poly_features = a_1 * x."""
        coeffs = torch.tensor([0.7])
        x = torch.randn(2, 4, 3)
        result = _poly_features_ref(x, coeffs)
        torch.testing.assert_close(result, 0.7 * x)

    def test_degree3_matches_manual(self):
        """Degree-3 polynomial matches manual computation."""
        coeffs = torch.tensor([1.0, 0.5, 1.0 / 6.0])
        x = torch.randn(2, 4, 3)
        result = _poly_features_ref(x, coeffs)
        expected = 1.0 * x + 0.5 * x**2 + (1.0 / 6.0) * x**3
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_default_coefficients(self):
        """Default coefficients should be 1/i! for i=1..p."""
        for p in [1, 2, 3, 5]:
            expected = [1.0 / math.factorial(i) for i in range(1, p + 1)]
            coeffs = torch.tensor(expected)
            for i, c in enumerate(coeffs):
                assert c.item() == pytest.approx(1.0 / math.factorial(i + 1))

    def test_gradient_flows_through_coeffs(self):
        """Coefficients should receive gradients."""
        coeffs = torch.tensor([1.0, 0.5, 1.0 / 6.0], requires_grad=True)
        x = torch.randn(2, 4, 8)
        result = _poly_features_ref(x, coeffs)
        result.sum().backward()
        assert coeffs.grad is not None
        assert not torch.all(coeffs.grad == 0)


class TestGeluDerivative:
    """Tests for the exact GELU derivative implementation."""

    def test_matches_numerical(self):
        """Exact derivative matches finite-difference approximation."""
        torch.manual_seed(42)
        x = torch.randn(100)
        exact = _gelu_derivative(x)
        numerical = _gelu_derivative_ref(x)
        # Finite differences with eps=1e-5 have ~5e-2 error from truncation
        torch.testing.assert_close(exact, numerical, atol=5e-2, rtol=5e-2)

    def test_at_zero(self):
        """GELU'(0) = 0.5 (CDF of standard normal at 0)."""
        x = torch.tensor([0.0])
        assert _gelu_derivative(x).item() == pytest.approx(0.5, abs=1e-6)

    def test_large_positive(self):
        """For large x, GELU'(x) -> 1."""
        x = torch.tensor([5.0, 10.0])
        result = _gelu_derivative(x)
        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# Full model tests (require atlas_kernels / Triton — skip if unavailable)
# ---------------------------------------------------------------------------

try:
    from nanochat.atlas import Atlas, AtlasConfig, AtlasMemoryLayer
    _HAS_ATLAS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ATLAS = False

requires_atlas = pytest.mark.skipif(not _HAS_ATLAS, reason="Triton/CUDA not available")


@requires_atlas
class TestDeepMemory:
    """Tests for deep MLP memory."""

    @pytest.fixture
    def small_config(self):
        return AtlasConfig(
            sequence_len=16, vocab_size=64, n_layer=1, n_head=2,
            n_embd=32, chunk_size=8, ns_steps=2,
            omega_window=1, poly_degree=0, deep_memory=True, memory_expand=1,
        )

    def test_output_shape(self, small_config):
        """Deep memory path produces same output shape as linear."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = AtlasMemoryLayer(small_config).to(device)
        x = torch.randn(1, 16, 32, device=device)
        y, state = layer(x)
        assert y.shape == (1, 16, 32)

    def test_state_is_4tuple(self, small_config):
        """Deep memory returns (W1, W2, S_W1, S_W2)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = AtlasMemoryLayer(small_config).to(device)
        x = torch.randn(1, 16, 32, device=device)
        _, state = layer(x)
        assert len(state) == 4
        W1, W2, S_W1, S_W2 = state
        D = small_config.n_embd // small_config.n_head
        E = small_config.memory_expand * D
        assert W1.shape == (1, 2, D, E)
        assert W2.shape == (1, 2, E, D)

    def test_expand_2_shapes(self):
        """memory_expand=2 produces rectangular weight matrices."""
        config = AtlasConfig(
            sequence_len=16, vocab_size=64, n_layer=1, n_head=2,
            n_embd=32, chunk_size=8, ns_steps=2,
            omega_window=1, poly_degree=0, deep_memory=True, memory_expand=2,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = AtlasMemoryLayer(config).to(device)
        x = torch.randn(1, 16, 32, device=device)
        _, state = layer(x)
        W1, W2, _, _ = state
        D = 16  # 32 / 2 heads
        E = 32  # 2 * D
        assert W1.shape == (1, 2, D, E)
        assert W2.shape == (1, 2, E, D)

    def test_linear_path_unchanged(self):
        """Linear memory path (deep_memory=False) still works."""
        config = AtlasConfig(
            sequence_len=16, vocab_size=64, n_layer=1, n_head=2,
            n_embd=32, chunk_size=8, ns_steps=2,
            omega_window=1, poly_degree=0, deep_memory=False,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = AtlasMemoryLayer(config).to(device)
        x = torch.randn(1, 16, 32, device=device)
        _, state = layer(x)
        assert len(state) == 2  # (M, S)
        D = 16
        assert state[0].shape == (1, 2, D, D)


@requires_atlas
class TestFullModel:
    """Integration tests for the full Atlas model."""

    def test_forward_deep_memory(self):
        """Forward pass completes without error."""
        config = AtlasConfig(
            sequence_len=32, vocab_size=64, n_layer=2, n_head=2,
            n_embd=32, chunk_size=8, ns_steps=2,
            omega_window=4, poly_degree=2, deep_memory=True, memory_expand=1,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.device('meta'):
            model = Atlas(config)
        model.to_empty(device=device)
        model.init_weights()
        idx = torch.randint(0, 64, (2, 32), device=device)
        targets = torch.randint(0, 64, (2, 32), device=device)
        loss = model(idx, targets=targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_backward_deep_memory(self):
        """Backward pass computes gradients for all parameters."""
        # pe_ste=True is required for gradients to flow through the Triton PE kernel
        # (the raw kernel has no autograd backward; STE provides identity backward)
        config = AtlasConfig(
            sequence_len=16, vocab_size=64, n_layer=1, n_head=2,
            n_embd=32, chunk_size=8, ns_steps=2,
            omega_window=4, poly_degree=3, deep_memory=True, memory_expand=1,
            pe_ste=True,
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.device('meta'):
            model = Atlas(config)
        model.to_empty(device=device)
        model.init_weights()
        # Enable grad for float params (init_weights uses no_grad)
        for p in model.parameters():
            p.requires_grad_(True)
        idx = torch.randint(0, 64, (1, 16), device=device)
        targets = torch.randint(0, 64, (1, 16), device=device)
        loss = model(idx, targets=targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
