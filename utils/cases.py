from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

CASES = {
    0: Case("Zflat_Hone_Vzero", 0.20),
    1: Case("Zflat_Hgauss_Vzero", 0.40),
    2: Case("Zflat_Hgauss0_Vzero", 0.40),
    3: Case("Zflat_Hsine_Vzero", 0.20),

    4: Case("Zinclined_Hone_Vzero", 0.20),
    5: Case("Zinclined_Hgauss_Vzero", 0.20),

    6: Case("Zflat_Hgauss_Vgauss", 0.25),

    7: Case("Zwavebreaker_Hgauss_Vzero", 0.4),
    8: Case("Zwavebreaker_Hgauss0_Vzero", 0.4),

    9: Case("Zwavebreaker_Hgauss_Vsine", 0.25),
    10: Case("Ztwowavebreakers_Hgauss_Vsine", 0.25),
}

def parse_cases(case_str: str) -> tuple[str, str, str]:
    z_case, h_case, v_case = case_str.split("_")

    z_case = z_case[1:]  # "Zflat" -> "flat"
    h_case = h_case[1:]  # "Hsine" -> "sine"
    v_case = v_case[1:]  # "Vsine" -> "sine"

    return z_case, h_case, v_case