from config import DEVICE, DTYPE

import torch

def get_x(a: float = 0.0, b: float = 1.0, nx: int = 256) -> torch.Tensor:
    return torch.linspace(a, b, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)


def get_t(T: float = 1.0, nt: int = 256) -> torch.Tensor:
    return torch.linspace(0.0, T, nt, device=DEVICE, dtype=DTYPE)
