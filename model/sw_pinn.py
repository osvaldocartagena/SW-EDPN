import torch
import torch.nn as nn

from model.mlp import MLP
from utils import CASES, parse_cases, get_topography, get_height, get_velocity


class SW_PINN(nn.Module):
    def __init__(self, case_id: int, width: int = 64, depth: int = 4):
        super().__init__()
        self.case_id = case_id
        self.case = CASES[case_id]
        self.z_case, self.h_case, self.v_case = parse_cases(self.case.name)
        self.net = MLP(width=width, depth=depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0, u0 = initial_condition(x, self.z_case, self.h_case, self.v_case)

        raw = self.net(x, t / self.case.T)
        nn_h = raw[:, 0:1]
        nn_u = raw[:, 1:2]

        envelope = t * x * (1.0 - x)

        h = h0 + envelope * nn_h
        u = u0 + envelope * nn_u

        return h, u


def initial_condition(x: torch.Tensor, z_case: str, h_case: str, v_case: str) -> tuple[torch.Tensor, torch.Tensor]:
    h0 = get_height(x, h_case)
    u0 = get_velocity(x, v_case)
    z = get_topography(x, z_case)

    eta = h0 - z

    return eta, u0
