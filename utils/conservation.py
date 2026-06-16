from typing import TYPE_CHECKING
from utils.domain import get_x, get_t
from utils.topography import get_topography

import torch

if TYPE_CHECKING:
    from model.sw_pinn import SW_PINN

G = 9.81  # m/s^2

def get_mass(model: "SW_PINN") -> dict[float, float]:
    """
    Masa total M(t) = ∫ h(x,t) dx
    """
    x = get_x().reshape(-1)
    times = get_t(model.case.T)
    mass: dict[float, float] = {}

    with torch.no_grad():
        for tau in times.reshape(-1):
            t = torch.full_like(x, float(tau))
            h, _ = model(x.unsqueeze(1), t.unsqueeze(1))
            h = h.reshape(-1)
            m = torch.trapz(h, x)
            mass[float(tau)] = float(m)

    return mass


def get_energy(model: "SW_PINN", g: float = G) -> dict[float, float]:
    """
    Energía mecánica total (por unidad de ancho) en SW 1D con topografía,
    medida respecto a un datum fijo (consistente con η = h + z):

        E(t) = ∫ [ 1/2 * h * u^2 + 1/2 * g * h^2 + g * h * z(x) ] dx

    Términos:
      - 1/2 * h * u^2  : energía cinética por unidad de área
      - 1/2 * g * h^2  : energía potencial relativa al fondo
      - g * h * z      : corrección por elevación del fondo (datum absoluto)
    """
    x = get_x().reshape(-1)
    times = get_t(model.case.T)
    z = get_topography(x, model.z_case).reshape(-1)  # z(x), no depende de t
    energy: dict[float, float] = {}

    with torch.no_grad():
        for tau in times.reshape(-1):
            t = torch.full_like(x, float(tau))
            h, u = model(x.unsqueeze(1), t.unsqueeze(1))
            h = h.reshape(-1)
            u = u.reshape(-1)

            e_density = 0.5 * h * u.pow(2) + 0.5 * g * h.pow(2) + g * h * z
            E = torch.trapz(e_density, x)
            energy[float(tau)] = float(E)

    return energy
