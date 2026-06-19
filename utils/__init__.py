# cases.py es puro stdlib y lo necesita tanto el PINN como el FVM.
from .cases import Case, CASES, parse_cases

# Los modulos siguientes dependen de torch (PINN). El FVM (que solo usa numpy)
# no los necesita; los hacemos opcionales para poder correr scripts/run_fvm.py
# sin tener torch instalado.
try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from .topography import get_topography
    from .free_surface import get_free_surface
    from .velocity import get_velocity
    from .plot import make_plots
    from .domain import get_x, get_t