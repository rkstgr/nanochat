"""H100 benchmark: compare PE approaches in full Atlas training context."""
import torch
import time
from nanochat.atlas import Atlas, AtlasConfig, AtlasMemoryLayer
from nanochat.atlas_kernels import (
    _polar_express_pytorch,
    fused_polar_express,
    compiled_polar_express,
    triton_autograd_polar_express,
)

device = 'cuda'
torch.manual_seed(42)

# Match H100 job config
config = AtlasConfig(
    sequence_len=2048, vocab_size=32768,
    n_layer=8, n_head=4, n_embd=512, chunk_size=64, ns_steps=5,
)
D = config.n_embd // config.n_head  # 128

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"D={D}, n_head={config.n_head}, n_layer={config.n_layer}")
print()

# === Microbenchmark: PE forward+backward with realistic shapes ===
B_pe = 16 * 64 * 4  # batch * chunk_size * n_head = 4096
print(f"=== PE microbenchmark: batch={B_pe}, D={D}, steps=5 ===")

def bench_pe(name, fn, N=30, warmup=10):
    X = torch.randn(B_pe, D, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    for _ in range(warmup):
        y = fn(X, 5)
        y.sum().backward()
        X.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        y = fn(X, 5)
        y.sum().backward()
        X.grad = None
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"  {name:45s} {ms:8.2f} ms")
    return ms

t_ref = bench_pe("Reference (PyTorch)", _polar_express_pytorch)
t_a = bench_pe("Approach A (compiled reduce-overhead)", compiled_polar_express)
# Approach B (triton autograd) fails on H100: backward kernel exceeds 228KB shared memory
# (needs ~328KB for all intermediate 128×128 matrices in the backward pass)

print(f"\n  Speedup: A={t_ref/t_a:.2f}x over reference")

# === Full model training step benchmark ===
print(f"\n=== Full model training step ===")

with torch.device('meta'):
    model = Atlas(config)
model = model.to_empty(device=device)
model.init_weights()

B = 16
x = torch.randint(0, config.vocab_size, (B, config.sequence_len), device=device)
targets = torch.randint(0, config.vocab_size, (B, config.sequence_len), device=device)

def bench_model(name, pe_fn, N=5, warmup=2):
    # Monkey-patch _process_chunk to use the given PE function
    import nanochat.atlas as atlas_mod
    original = atlas_mod.fused_polar_express

    # Replace in the module namespace that _process_chunk closes over
    atlas_mod.fused_polar_express = pe_fn

    # Also need to update the import in atlas.py
    from nanochat.atlas_kernels import fused_polar_express as _orig_kernel_fn
    import nanochat.atlas_kernels as kernels_mod
    kernels_mod_orig = kernels_mod.fused_polar_express

    # The _process_chunk is compiled — we need to recompile with new PE
    # Actually _process_chunk references fused_polar_express from the import,
    # so let's just patch atlas_kernels.fused_polar_express
    kernels_mod.fused_polar_express = pe_fn

    for _ in range(warmup):
        loss = model(x, targets)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        loss = model(x, targets)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000

    # Restore
    kernels_mod.fused_polar_express = kernels_mod_orig
    atlas_mod.fused_polar_express = original

    tokens = B * config.sequence_len
    print(f"  {name:45s} {ms/1000:8.2f} s  ({tokens/(ms/1000):,.0f} tok/s)")
    return ms

# Test with each approach
# Note: the model uses fused_polar_express which on H100 dispatches to Triton fwd kernel
# We'll replace it with each approach for comparison

t_current = bench_model("Current (fused Triton fwd, autograd bwd)", fused_polar_express)
t_model_a = bench_model("Approach A (compiled reduce-overhead)", compiled_polar_express)

print(f"\n  Model speedup: A={t_current/t_model_a:.2f}x over current")
print(f"  Winner: {'A (compiled)' if t_model_a < t_current else 'Current (fused Triton fwd)'}")
