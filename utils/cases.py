from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

# Nomenclatura del string del caso:
#   Z<topografia>_S<superficie>_V0<velocidad-inicial>
# Ejemplos: "Zflat_Sone_V0zero", "Zwavebreaker_Sgauss_V0zero".
# Los parametros se extraen via parse_cases().
CASES = {
    0: Case("Zflat_Sone_V0zero",         1.0),
    1: Case("Zflat_Sgauss_V0zero",       1.0),
    2: Case("Zwavebreaker_Sgauss_V0zero", 1.0),
    3: Case("Zwavebreaker_Sgauss0_V0zero", 0.4),
}

# Prefijos del string del caso. Cualquier nombre nuevo debe respetarlos.
_PREFIXES = {"z": "Z", "s": "S", "v0": "V0"}


def parse_cases(case_str: str) -> tuple[str, str, str]:
    """Extrae (z_case, s_case, v0_case) del nombre estandar.

    Ejemplo: "Zflat_Sgauss_V0zero" -> ("flat", "gauss", "zero").
    """
    parts = case_str.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Nombre de caso invalido '{case_str}': se esperan 3 partes "
            f"separadas por '_' con prefijos Z/S/V0."
        )
    z_part, s_part, v0_part = parts

    for label, part, prefix in (
        ("z", z_part, _PREFIXES["z"]),
        ("s", s_part, _PREFIXES["s"]),
        ("v0", v0_part, _PREFIXES["v0"]),
    ):
        if not part.startswith(prefix):
            raise ValueError(
                f"Parte '{label}' = '{part}' no empieza con prefijo '{prefix}'."
            )

    return (
        z_part[len(_PREFIXES["z"]):],
        s_part[len(_PREFIXES["s"]):],
        v0_part[len(_PREFIXES["v0"]):],
    )
