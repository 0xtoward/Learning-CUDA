import torch

atol = 2.5e-2
rtol = 3.0e-3


def reference(A, B, C, M, N, K, **_):
    # Correctness reference: strict FP32 arithmetic over the exact FP16 inputs.
    a = A[: M * K].view(M, K).float()
    b = B[: K * N].view(K, N).float()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        C[: M * N].copy_((a @ b).reshape(-1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
