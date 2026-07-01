from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

# Case name convention:
#   Z<topography>_S0<surface>_V0<initial-velocity>
# Examples: "Zflat_S0one_V0zero", "Zwavebreaker_S0gauss_V0zero".
# Components are extracted via parse_cases().
CASES = {
    0: Case("Zflat_S0one_V0zero",          0.4),
    1: Case("Zflat_S0gauss_V0zero",        0.4),
    2: Case("Zwavebreaker_S0gauss_V0zero", 0.4),
}

# Prefixes used in the case name string. New cases must respect these.
_PREFIXES = {"z": "Z", "s0": "S0", "v0": "V0"}


def parse_cases(case_str: str) -> tuple[str, str, str]:
    """Extrae (z_case, s0_case, v0_case) del nombre estandar.

    Ejemplo: "Zflat_S0gauss_V0zero" -> ("flat", "gauss", "zero").
    """
    parts = case_str.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid case name '{case_str}': expected 3 parts "
            f"separated by '_' with prefixes Z/S0/V0."
        )
    z_part, s0_part, v0_part = parts

    for label, part, prefix in (
        ("z", z_part, _PREFIXES["z"]),
        ("s0", s0_part, _PREFIXES["s0"]),
        ("v0", v0_part, _PREFIXES["v0"]),
    ):
        if not part.startswith(prefix):
            raise ValueError(
                f"Part '{label}' = '{part}' does not start with prefix '{prefix}'."
            )

    return (
        z_part[len(_PREFIXES["z"]):],
        s0_part[len(_PREFIXES["s0"]):],
        v0_part[len(_PREFIXES["v0"]):],
    )


def display_label(case_name: str) -> str:
    """The case name already serves as the display label.

    Kept for import compatibility; returns the case name unchanged.
    """
    return case_name

