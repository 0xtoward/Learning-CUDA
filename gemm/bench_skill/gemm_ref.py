import torch

# TF32 cuBLAS deliberately trades mantissa precision for Tensor Core speed.
atol = 5e-3
rtol = 5e-3


def reference(A, B, C, M, N, K, **_):
    a = A[: M * K].view(M, K)
    b = B[: K * N].view(K, N)
    C[: M * N].copy_((a @ b).reshape(-1))

