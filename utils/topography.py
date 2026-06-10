import torch

def get_topography(x: torch.Tensor, z_case: str) -> torch.Tensor:
    match z_case:
        case "flat":
            return flat_topography(x)
        case "inclined":
            return inclined_topography(x)
        case "cos":
            return cos_topography(x)        
        case "wavebreaker":
            return wavebreaker_topography(x)    
        case _:
            raise ValueError(f"Topografía inválida: {z_case}")
        
def wavebreaker_topography(x: torch.Tensor) -> torch.Tensor:
    a = 0.4
    b = 0.6
    H = 1.2
    k = 10.0
    return (H/2) * (torch.tanh(k * (x - a)) - torch.tanh(k * (x - b)))  

def cos_topography(x: torch.Tensor) -> torch.Tensor:       
    return 0.05 * torch.cos(2.0 * torch.pi * x)

def inclined_topography(x: torch.Tensor) -> torch.Tensor:
    return 0.05 * x

def flat_topography(x: torch.Tensor) -> torch.Tensor:
    return 0.0 * x  # conserva el grafo para poder hacer grad(z,x)
