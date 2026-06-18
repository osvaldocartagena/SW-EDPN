import torch


def get_free_surface(x: torch.Tensor, h_case: str) -> torch.Tensor:
    """
    Devuelve la elevación inicial de la superficie libre η(x, t=0).

    Convención: η = h + z, donde h es la profundidad de la columna de agua
    y z es la elevación del fondo. La profundidad inicial se recupera como
    h0 = η(x, 0) - z(x).
    """
    match h_case:
        case "sine":
            return sine_free_surface(x)
        case "gauss":
            return gauss_free_surface(x)
        case "gauss0":
            return gauss0_free_surface(x)
        case "one":
            return one_free_surface(x)
        case _:
            raise ValueError(f"Superficie libre inicial inválida: {h_case}")


def gauss_free_surface(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.2 * torch.exp(-120.0 * (x - 0.2) ** 2)

def gauss0_free_surface(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.2 * torch.exp(-120.0 * (x) ** 2)

def sine_free_surface(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.2 * torch.sin(torch.pi * x)


def one_free_surface(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + 0.0 * x
