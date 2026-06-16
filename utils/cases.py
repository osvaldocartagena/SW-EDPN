from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

CASES = {
    0: Case("Zflat_Hone_Vzero", 0.20),
    1: Case("Zflat_Hgauss_Vzero", 0.20),

    2: Case("Zinclined_Hone_Vzero", 0.20),
    3: Case("Zinclined_Hgauss_Vzero", 0.20),

    4: Case("Zflat_Hgauss_Vgauss", 0.25),
    5: Case("Zwavebreaker_Hgauss_Vzero", 0.25),

    6: Case("Zwavebreaker_Hgauss_Vsine", 0.25),
    7: Case("Ztwowavebreakers_Hgauss_Vsine", 0.25),
}

def parse_cases(case_str: str) -> tuple[str, str, str]:
    z_case, h_case, v_case = case_str.split("_")

    z_case = z_case[1:]  # "Zflat" -> "flat"
    h_case = h_case[1:]  # "Hsine" -> "sine"
    v_case = v_case[1:]  # "Vsine" -> "sine"

    return z_case, h_case, v_case