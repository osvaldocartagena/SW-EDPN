from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

CASES = {
    0: Case("Zflat_Hsine_Vzero", 0.20),
    1: Case("Zflat_Hsine_Vsine", 0.20),
    2: Case("Zflat_Hgauss_Vzero", 0.25),
    3: Case("Zflat_Hone_Vzero", 0.20),
    4: Case("Zinclined_Hone_Vzero", 0.15),
    5: Case("Zwavebreaker_Hgauss_Vzero", 0.25),
}


# ----------------------------
# Topografia e IC
# ----------------------------

def parse_cases(case_str: str) -> tuple[str, str, str]:
    z_case, h_case, v_case = case_str.split("_")

    z_case = z_case[1:]  # "Zflat" -> "flat"
    h_case = h_case[1:]   # "Hsine" -> "sine
    v_case = v_case[1:]   # "Vsine" -> "sine    

    return z_case, h_case, v_case