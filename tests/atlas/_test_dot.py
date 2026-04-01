import triton, triton.language as tl, torch

@triton.jit
def _test_dot(A_ptr, B_ptr, C_ptr, D: tl.constexpr):
    offs_i = tl.arange(0, D)
    offs_j = tl.arange(0, D)
    a = tl.load(A_ptr + offs_i[:, None] * D + offs_j[None, :])
    b = tl.load(B_ptr + offs_i[:, None] * D + offs_j[None, :])
    c = tl.dot(a, b)
    tl.store(C_ptr + offs_i[:, None] * D + offs_j[None, :], c)

D = 64
A = torch.randn(D, D, device='cuda', dtype=torch.float32)
B = torch.randn(D, D, device='cuda', dtype=torch.float32)
C = torch.empty(D, D, device='cuda', dtype=torch.float32)
_test_dot[(1,)](A, B, C, D=D)
ref = A @ B
print(f'D={D} tl.dot max diff: {(C - ref).abs().max().item():.2e}')
print('OK')
