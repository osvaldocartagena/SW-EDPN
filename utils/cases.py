from dataclasses import dataclass

@dataclass
class Case:
    name: str
    T: float

CASES = {
    0: Case("Zflat_Hsine_Vzero", 0.20),
    1: Case("Zflat_Hsine_Vsine", 0.20),
    2: Case("flat_gaussian_bump", 0.15),
    3: Case("quiet_flat", 0.20),
    4: Case("slope_lake_at_rest", 0.15),
    5: Case("bottom_bump_perturbation", 0.15),
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