import torch

atol = 0.1
rtol = 1.0e-4


def reference(x, out, n, **_):
    out[0] = torch.sum(x[:n], dtype=torch.float32)
