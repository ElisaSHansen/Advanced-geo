import math
from dataclasses import dataclass

# ============================================================
# Geometry + steel
# ============================================================
def tube_section(D: float, t: float) -> dict:
    """
    Circular tube:
      d = D - 2t
      I = π/64 (D^4 - d^4)
      W = I / (D/2)   (elastic section modulus)
    """
    d = D - 2.0 * t
    if d <= 0:
        raise ValueError("Inner diameter <= 0. Check D and t.")
    I = math.pi / 64.0 * (D**4 - d**4)
    W = I / (D / 2.0)
    return {"D": D, "t": t, "d": d, "I": I, "W": W}

def steel_yield_moment(fy_MPa: float, W_m3: float) -> float:
    """
    My = fy * W
    fy: MPa -> kN/m^2 using 1 MPa = 1000 kN/m^2
    returns My in kN*m
    """
    fy_kN_m2 = fy_MPa * 1000.0
    return fy_kN_m2 * W_m3

# ============================================================
# Rankine Kp
# ============================================================
def rankine_Kp(phi_deg: float) -> float:
    phi = math.radians(phi_deg)
    return math.tan(math.radians(45.0) + phi / 2.0) ** 2


# ============================================================
# Layer averaging (exclude disturbed zone if desired)
# ============================================================
@dataclass
class Layer:
    z_top: float      # m below seabed
    z_bot: float      # m below seabed
    gamma: float      # kN/m^3 (total/sat)
    phi: float | None # deg

def weighted_avg_over_interval(layers: list[Layer], z0: float, z1: float) -> dict:
    """
    Thickness-weighted averages over [z0, z1] (m below seabed).
    Only includes overlapping portions.
    """
    if z1 <= z0:
        raise ValueError("z1 must be > z0")

    thickness_sum = 0.0
    gamma_sum = 0.0
    phi_sum = 0.0
    phi_thickness_sum = 0.0

    for L in layers:
        a = max(z0, L.z_top)
        b = min(z1, L.z_bot)
        if b <= a:
            continue
        dz = b - a
        thickness_sum += dz
        gamma_sum += L.gamma * dz
        if L.phi is not None:
            phi_sum += L.phi * dz
            phi_thickness_sum += dz

    if thickness_sum <= 0:
        raise RuntimeError("No overlap between layers and averaging interval.")

    gamma_avg = gamma_sum / thickness_sum
    phi_avg = (phi_sum / phi_thickness_sum) if phi_thickness_sum > 0 else None

    return {"gamma_avg": gamma_avg, "phi_avg": phi_avg, "thickness": thickness_sum}


# ============================================================
# Broms (1964) - FLEXIBLE, free head, sand (yield-governed)
#   Hu = (3/2) * gamma' * D * Kp * f^2
#   Mmax = Hu * ( e + (2/3) f ) = My
# ============================================================
@dataclass
class BromsSandFlexibleInputs:
    D: float
    Lemb: float
    phi_deg: float
    gamma_total: float
    e: float
    fy_MPa: float = 355.0
    t_rule: str = "D/100"
    use_submerged_gamma: bool = True
    gamma_w: float = 9.81

def broms_sand_flexible_free_head(inp: BromsSandFlexibleInputs) -> dict:
    # thickness from rule
    if inp.t_rule.lower() == "d/100":
        t = inp.D / 100.0
    else:
        raise ValueError("Unsupported t_rule. Use 'D/100'.")

    # gamma'
    gamma_used = inp.gamma_total - inp.gamma_w if inp.use_submerged_gamma else inp.gamma_total
    if gamma_used <= 0:
        raise ValueError("gamma_used <= 0. Check gamma_total/gamma_w.")

    # Kp
    Kp = rankine_Kp(inp.phi_deg)

    # My (elastic)
    sec = tube_section(inp.D, t)
    My_kNm = steel_yield_moment(inp.fy_MPa, sec["W"])  # kN*m

    # Hu = A f^2, A = (3/2) gamma' D Kp
    A = 1.5 * gamma_used * inp.D * Kp

    def f_from_Hu(Hu_kN: float) -> float:
        return math.sqrt(Hu_kN / A)

    def residual(Hu_kN: float) -> float:
        f = f_from_Hu(Hu_kN)
        return Hu_kN * (inp.e + (2.0 / 3.0) * f) - My_kNm

    # bracket + bisection
    Hu_lo, Hu_hi = 1e-6, 1e6
    r_lo, r_hi = residual(Hu_lo), residual(Hu_hi)
    it = 0
    while r_lo * r_hi > 0 and it < 80:
        Hu_hi *= 2.0
        r_hi = residual(Hu_hi)
        it += 1
    if r_lo * r_hi > 0:
        raise RuntimeError("Could not bracket Hu root. Check inputs.")

    for _ in range(120):
        Hu_mid = 0.5 * (Hu_lo + Hu_hi)
        r_mid = residual(Hu_mid)
        if r_lo * r_mid <= 0:
            Hu_hi, r_hi = Hu_mid, r_mid
        else:
            Hu_lo, r_lo = Hu_mid, r_mid

    Hu_kN = 0.5 * (Hu_lo + Hu_hi)
    f = f_from_Hu(Hu_kN)
    Mmax_kNm = Hu_kN * (inp.e + (2.0 / 3.0) * f)

    # slide approximation: f ≈ 0.82*sqrt(Hu/(gamma' D Kp))
    f_slide = 0.82 * math.sqrt(Hu_kN / (gamma_used * inp.D * Kp))

    return {
        "t_m": t,
        "Kp": Kp,
        "gamma_prime_kN_m3": gamma_used,
        "My_MNm": My_kNm / 1000.0,
        "Hu_MN": Hu_kN / 1000.0,
        "f_m": f,
        "f_slide_m": f_slide,
        "Mmax_MNm": Mmax_kNm / 1000.0,
        "f_within_Lemb": (f <= inp.Lemb),
    }


# ============================================================
# Broms (slide) - RIGID, free head, sand (soil mechanism)
#   Hu_rigid = 0.5 * gamma' * D^3 * Kp / (e + Lemb)
#   f_rigid  = 0.82 * sqrt( Hu_rigid / (gamma' D Kp) )
#   Mmax_rigid = Hu_rigid * ( e + 2/3 f_rigid )
# ============================================================
def broms_sand_rigid_free_head(
    *,
    D: float,
    Lemb: float,
    e: float,
    phi_deg: float,
    gamma_total: float,
    fy_MPa: float = 355.0,
    t_rule: str = "D/100",
    use_submerged_gamma: bool = True,
    gamma_w: float = 9.81,
) -> dict:
    if t_rule.lower() == "d/100":
        t = D / 100.0
    else:
        raise ValueError("Unsupported t_rule. Use 'D/100'.")

    gamma_prime = gamma_total - gamma_w if use_submerged_gamma else gamma_total
    if gamma_prime <= 0:
        raise ValueError("gamma_prime <= 0. Check gamma_total/gamma_w.")

    Kp = rankine_Kp(phi_deg)

    sec = tube_section(D, t)
    My_kNm = steel_yield_moment(fy_MPa, sec["W"])

    Hu_rigid_kN = 0.5 * gamma_prime * (D**3) * Kp / (e + Lemb)
    f_rigid_m = 0.82 * math.sqrt(Hu_rigid_kN / (gamma_prime * D * Kp))
    Mmax_rigid_kNm = Hu_rigid_kN * (e + (2.0 / 3.0) * f_rigid_m)

    return {
        "t_m": t,
        "Kp": Kp,
        "gamma_prime_kN_m3": gamma_prime,
        "My_MNm": My_kNm / 1000.0,
        "Hu_rigid_MN": Hu_rigid_kN / 1000.0,
        "f_rigid_m": f_rigid_m,
        "Mmax_rigid_MNm": Mmax_rigid_kNm / 1000.0,
        "rigid_moment_ok": (My_kNm >= Mmax_rigid_kNm),
    }


# ============================================================
# Group sizing
# ============================================================
def ceil_int(x: float) -> int:
    return int(math.ceil(x))

def check_group(
    Px_total_MN: float,
    e_m: float,
    Pv_total_MN: float,
    Qult_per_pile_MN: float,
    Hu_per_pile_MN: float,
    n_from_part2: int | None = None
) -> dict:
    n_req_ax = ceil_int(Pv_total_MN / Qult_per_pile_MN) if Qult_per_pile_MN > 0 else None
    n_req_lat = ceil_int(Px_total_MN / Hu_per_pile_MN) if Hu_per_pile_MN > 0 else None

    n_use = n_from_part2 if n_from_part2 is not None else (n_req_ax if n_req_ax is not None else 1)

    Px_per_pile = Px_total_MN / n_use
    Pv_per_pile = Pv_total_MN / n_use
    M_per_pile = Px_per_pile * e_m  # MNm

    return {
        "n_use": n_use,
        "n_req_axial": n_req_ax,
        "n_req_lateral": n_req_lat,
        "Px_per_pile_MN": Px_per_pile,
        "Pv_per_pile_MN": Pv_per_pile,
        "M_per_pile_MNm": M_per_pile,
        "axial_ok": (Qult_per_pile_MN >= Pv_per_pile),
        "lateral_ok": (Hu_per_pile_MN >= Px_per_pile),
        "n_governing_min": max(n for n in [n_req_ax, n_req_lat] if n is not None),
    }


# ============================================================
# Clean printing helpers
# ============================================================
def yn(flag: bool) -> str:
    return "YES" if flag else "NO"

def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"

def print_block(title: str, rows: list[tuple[str, str]], width: int = 66) -> None:
    print("\n" + "=" * width)
    print(title)
    print("-" * width)
    for k, v in rows:
        print(f"{k:<38s} {v:>26s}")
    print("=" * width)


if __name__ == "__main__":
    # -----------------------------
    # GROUP 19 (example inputs)
    # -----------------------------
    group = 19
    D = 2.5
    Lemb = 16.0
    e = 23.3

    # Choose horizontal load used for lateral pile count:
    # dataset: Px=6.5 MN, assignment example: Pn=3 MN
    Px_dataset_MN = 6.5
    Pn_assignment_MN = 3.0
    Px_total_MN = Px_dataset_MN  # <-- swap to Pn_assignment_MN if needed

    # Disturbed zone exclusion: average from z0 to z1
    z0 = 3.3
    z1 = Lemb

    # Soil layers (sand) - valid as long as Lemb is within these layers
    layers = [
        Layer(3.3, 13.6, gamma=20.47, phi=47.55),
        Layer(13.6, 25.0, gamma=19.24, phi=36.81),
        Layer(25.0, 28.7, gamma=18.00, phi=31.68),
    ]

    av = weighted_avg_over_interval(layers, z0, z1)
    gamma_total = av["gamma_avg"]
    phi_deg = av["phi_avg"]
    use_submerged_gamma = True

    # Placeholders (update when Part 2 is done)
    Pv_total_MN = 155.0
    Qult_per_pile_MN = 51.69
    n_from_part2 = 3 # set to your Part 2 pile count when ready

    # Rigid + Flexible
    rigid = broms_sand_rigid_free_head(
        D=D, Lemb=Lemb, e=e, phi_deg=phi_deg, gamma_total=gamma_total,
        fy_MPa=355.0, t_rule="D/100", use_submerged_gamma=use_submerged_gamma, gamma_w=9.81
    )
    flex = broms_sand_flexible_free_head(
        BromsSandFlexibleInputs(
            D=D, Lemb=Lemb, phi_deg=phi_deg, gamma_total=gamma_total, e=e,
            fy_MPa=355.0, t_rule="D/100", use_submerged_gamma=use_submerged_gamma
        )
    )

    # Governing ultimate lateral capacity = max(rigid, flexible)
    Hu_governing_MN = max(rigid["Hu_rigid_MN"], flex["Hu_MN"])
    governing_mode = "RIGID (soil mechanism)" if rigid["Hu_rigid_MN"] > flex["Hu_MN"] else "FLEXIBLE (yield-governed)"

    # Group check using governing Hu
    checks = check_group(
        Px_total_MN=Px_total_MN,
        e_m=e,
        Pv_total_MN=Pv_total_MN,
        Qult_per_pile_MN=Qult_per_pile_MN,
        Hu_per_pile_MN=Hu_governing_MN,
        n_from_part2=n_from_part2
    )

    # -----------------------------
    # Clean output
    # -----------------------------
    print_block(
        f"GROUP {group} | INPUTS + SOIL AVERAGES",
        [
            ("Pile diameter D [m]", fmt(D, 2)),
            ("Embedded length Lemb [m]", fmt(Lemb, 2)),
            ("Eccentricity e [m]", fmt(e, 2)),
            ("Averaging interval [m]", f"{fmt(z0,2)} to {fmt(z1,2)} (thk {fmt(av['thickness'],2)})"),
            ("Weighted gamma_total [kN/m^3]", fmt(gamma_total, 3)),
            ("Weighted phi [deg]", fmt(phi_deg, 3)),
            ("Use submerged unit weight?", yn(use_submerged_gamma)),
            ("Horizontal load used Px_total [MN]", fmt(Px_total_MN, 3)),
        ],
    )

    print_block(
        "Broms RIGID check (free head, sand)",
        [
            ("Kp [-]", fmt(rigid["Kp"], 3)),
            ("gamma' [kN/m^3]", fmt(rigid["gamma_prime_kN_m3"], 3)),
            ("t = D/100 [m]", fmt(rigid["t_m"], 4)),
            ("Hu_rigid [MN]", fmt(rigid["Hu_rigid_MN"], 6)),
            ("f_rigid [m]", fmt(rigid["f_rigid_m"], 3)),
            ("Mmax_rigid [MNm]", fmt(rigid["Mmax_rigid_MNm"], 3)),
            ("My [MNm]", fmt(rigid["My_MNm"], 3)),
            ("Moment check My >= Mmax_rigid?", yn(rigid["rigid_moment_ok"])),
        ],
    )

    print_block(
        "Broms FLEXIBLE solution (yield-governed, free head, sand)",
        [
            ("Kp [-]", fmt(flex["Kp"], 3)),
            ("gamma' [kN/m^3]", fmt(flex["gamma_prime_kN_m3"], 3)),
            ("t = D/100 [m]", fmt(flex["t_m"], 4)),
            ("Hu_flexible [MN]", fmt(flex["Hu_MN"], 3)),
            ("f [m]", fmt(flex["f_m"], 3)),
            ("f (slide approx) [m]", fmt(flex["f_slide_m"], 3)),
            ("Mmax [MNm]", fmt(flex["Mmax_MNm"], 3)),
            ("My [MNm]", fmt(flex["My_MNm"], 3)),
            ("f <= Lemb?", yn(flex["f_within_Lemb"])),
        ],
    )

    print_block(
        "GOVERNING ultimate lateral capacity (per pile)",
        [
            ("Hu_governing [MN]", fmt(Hu_governing_MN, 3)),
            ("Governing mode", governing_mode),
        ],
    )

    print_block(
        "PILE GROUP CHECK (equal distribution)",
        [
            ("Placeholder Pv_total [MN]", fmt(Pv_total_MN, 2)),
            ("Placeholder Qult per pile [MN]", fmt(Qult_per_pile_MN, 2)),
            ("Minimum n from axial", str(checks["n_req_axial"])),
            ("Minimum n from lateral", str(checks["n_req_lateral"])),
            ("n used for checks", str(checks["n_use"])),
            ("Per pile Px [MN]", fmt(checks["Px_per_pile_MN"], 3)),
            ("Per pile moment M = Px*e [MNm]", fmt(checks["M_per_pile_MNm"], 3)),
            ("Axial OK?", yn(checks["axial_ok"])),
            ("Lateral OK?", yn(checks["lateral_ok"])),
            ("Governing minimum n", str(checks["n_governing_min"])),
        ],
    )