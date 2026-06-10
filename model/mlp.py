import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        X = torch.cat([2.0 * x - 1.0, 2.0 * t - 1.0], dim=1)
        return self.net(X)