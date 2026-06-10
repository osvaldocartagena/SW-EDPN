import torch

def get_height(x: torch.Tensor, h_case: str) -> torch.Tensor:
    match h_case:
        case "sine":
            return sine_height(x)
        case "gauss":
            return gauss_height(x)
        case "one":
            return one_height(x)
        case _:
            raise ValueError(f"Altura inicial inválida: {h_case}")
        
def gauss_height(x: torch.Tensor) -> torch.Tensor:  
    return 1.0 + 0.2 * torch.exp(-120.0 * (x - 0.2) ** 2)

def sine_height(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.05 * torch.sin(torch.pi * x)

def one_height(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.0 * x