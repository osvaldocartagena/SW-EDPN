from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

# Nomenclatura del string del caso:
#   Z<topografia>_S0<superficie>_V0<velocidad-inicial>
# Ejemplos: "Zflat_S0one_V0zero", "Zwavebreaker_S0gauss_V0zero".
# Los parametros se extraen via parse_cases().
CASES = {
    0: Case("Zflat_S0one_V0zero",          0.4),
    1: Case("Zflat_S0gauss_V0zero",        0.4),
    2: Case("Zwavebreaker_S0gauss_V0zero", 0.4),
}

# Prefijos del string del caso. Cualquier nombre nuevo debe respetarlos.
_PREFIXES = {"z": "Z", "s0": "S0", "v0": "V0"}


def parse_cases(case_str: str) -> tuple[str, str, str]:
    """Extrae (z_case, s0_case, v0_case) del nombre estandar.

    Ejemplo: "Zflat_S0gauss_V0zero" -> ("flat", "gauss", "zero").
    """
    parts = case_str.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Nombre de caso invalido '{case_str}': se esperan 3 partes "
            f"separadas por '_' con prefijos Z/S0/V0."
        )
    z_part, s0_part, v0_part = parts

    for label, part, prefix in (
        ("z", z_part, _PREFIXES["z"]),
        ("s0", s0_part, _PREFIXES["s0"]),
        ("v0", v0_part, _PREFIXES["v0"]),
    ):
        if not part.startswith(prefix):
            raise ValueError(
                f"Parte '{label}' = '{part}' no empieza con prefijo '{prefix}'."
            )

    return (
        z_part[len(_PREFIXES["z"]):],
        s0_part[len(_PREFIXES["s0"]):],
        v0_part[len(_PREFIXES["v0"]):],
    )


def display_label(case_name: str) -> str:
    """Alias trivial: el case.name ya respeta el formato de display.

    Se mantiene esta funcion para no romper imports, pero como la
    nomenclatura interna y la de display coinciden, simplemente devuelve
    el nombre del caso sin modificacion.
    """
    return case_name

