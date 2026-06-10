import torch

def get_velocity(x: torch.Tensor, v_case: str) -> torch.Tensor:
    match v_case:
        case "sine":
            return sine_velocity(x)
        case "zero":
            return null_velocity(x)
        case _:
            raise ValueError(f"Velocidad inicial inválida: {v_case}")

def sine_velocity(x: torch.Tensor) -> torch.Tensor:     
    return 0.05 * torch.sin(torch.pi * x)

def null_velocity(x: torch.Tensor) -> torch.Tensor:
    return 0.0 * x  