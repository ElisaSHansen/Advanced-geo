import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CPT REPORT SCRIPT (Assignment 1 - Part 1)
# Computes:
#   qt, Rf, gamma, stresses, qn, Qt, Fr, Bq, Ic + soil zones
# Adds:
#   cu (undrained shear strength), phi' (friction angle), Vs (shear wave velocity)
# Adds stiffness Es (chosen):
#   - Zones 2–4 (clay/silt): gradual (3..8)*qc
#   - Zones 5–7 (sand): Es = (1 + Dr(z)^2) * qc
#       where Dr(z) computed by:
#         Dr = (1/C2) * ln( (qc/pa) / ( C0*(σ'v0/pa)^C1 ) )
#         C0=17.68, C1=0.5, C2=3.1, pa=100 kPa
#       Dr clipped to [0, 1]
#
# Units:
#   qc input MPa, fs kPa, u2 kPa
#   stresses kPa
#   Es stored in kPa, plotted in MPa
# Exports:
#   Clean Excel report
# Plots:
#   stresses, CPT panels, Ic chart, derived panels (cu/phi'/Vs/Es), Es comparison
# ============================================================

# ----------------------------
# USER SETTINGS
# ----------------------------
BASE_DIR = Path(__file__).parent
EXCEL_PATH = BASE_DIR / "cpt profile 1 (1).xlsx"
SHEET_NAME = "Data Sheet"

ZW_WATER_DEPTH_M = 22.0   # water depth (m)
GAMMA_W = 9.81            # kN/m^3  == kPa/m
PA_KPA = 100.0            # kPa (atmospheric pressure)
PA_MPA = PA_KPA / 1000.0  # MPa

# Strength
NKT = 17.0

# Dr correlation constants (from your figure)
DR_C0 = 17.68
DR_C1 = 0.5
DR_C2 = 3.1

OUT_EXCEL = BASE_DIR / "Part1_CPT_Report.xlsx"

# Optional: horizontal layer boundary lines in plots
LAYER_LINES_M = []  # e.g. [2.0, 6.0, 10.5]
SAVE_FIGS = True


# ----------------------------
# HELPERS: Reading
# ----------------------------
def read_cpt_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Reads CPT excel format where numeric data starts after header rows.
    Returns numeric df with columns:
      depth_m, qc_MPa, fs_kPa, u2_kPa, a
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")

    def is_num(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    start_idx = None
    for i in range(len(raw)):
        if is_num(raw.iloc[i, 0]) and is_num(raw.iloc[i, 1]) and is_num(raw.iloc[i, 2]):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find start of numeric CPT data in the sheet.")

    data = raw.iloc[start_idx:, :5].copy()
    data.columns = ["depth_m", "qc_MPa", "fs_kPa", "u2_kPa", "a"]
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["depth_m", "qc_MPa", "fs_kPa"]).sort_values("depth_m").reset_index(drop=True)

    # Fill missing u2/a
    if data["u2_kPa"].isna().all():
        data["u2_kPa"] = 0.0
    else:
        data["u2_kPa"] = data["u2_kPa"].fillna(0.0)

    if data["a"].isna().all():
        data["a"] = 1.0
    else:
        data["a"] = data["a"].fillna(1.0)

    return data


# ----------------------------
# HELPERS: Core CPT calculations
# ----------------------------
def compute_qt_MPa(qc_MPa: np.ndarray, u2_kPa: np.ndarray, a: np.ndarray) -> np.ndarray:
    # qt = qc + (1-a)u2 with consistent units (u2: kPa -> MPa)
    return qc_MPa + (1.0 - a) * (u2_kPa / 1000.0)


def compute_Rf_percent(fs_kPa: np.ndarray, qt_MPa: np.ndarray) -> np.ndarray:
    # Rf(%) = fs / qt *100 ; qt MPa -> kPa
    qt_kPa = qt_MPa * 1000.0
    qt_kPa = np.where(qt_kPa <= 1e-9, np.nan, qt_kPa)
    return (fs_kPa / qt_kPa) * 100.0


def estimate_gamma_kNm3(Rf_percent: np.ndarray, qt_MPa: np.ndarray) -> np.ndarray:
    """
    gamma/gamma_w = 0.27 log10(Rf) + 0.36 log10(qt/pa) + 1.236
    Rf in %, qt & pa in MPa, gamma_w in kN/m^3
    """
    Rf_safe = np.clip(Rf_percent, 0.1, None)
    qt_safe = np.clip(qt_MPa, 1e-9, None)
    gamma_ratio = 0.27 * np.log10(Rf_safe) + 0.36 * np.log10(qt_safe / PA_MPA) + 1.236
    return gamma_ratio * GAMMA_W


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * dx
    return out


def compute_stresses(depth_m: np.ndarray, gamma_total: np.ndarray):
    """
    Offshore (seabed is z=0):
      sigma_v0(z) = gamma_w*zw + ∫ gamma_total dz
      u0(z)       = gamma_w*(zw + z)
      sigma'_v0   = sigma_v0 - u0
    Returns kPa.
    """
    # Ensure seabed row at z=0 for stable integration
    if depth_m[0] > 1e-9:
        depth_m = np.insert(depth_m, 0, 0.0)
        gamma_total = np.insert(gamma_total, 0, gamma_total[0])

    sigma_water = GAMMA_W * ZW_WATER_DEPTH_M  # kPa
    sigma_soil = cumulative_trapz(gamma_total, depth_m)  # kPa
    sigma_v0 = sigma_water + sigma_soil
    u0 = GAMMA_W * (ZW_WATER_DEPTH_M + depth_m)
    sigma_v0_eff = sigma_v0 - u0

    return depth_m, sigma_v0, u0, sigma_v0_eff


# ----------------------------
# HELPERS: Ic zones
# ----------------------------
def ic_zone_and_sbt(ic: float):
    """
    Ic ranges used in your assignment:
      Zone 2: > 3.6  Organic soils - clay
      Zone 3: 2.95–3.6  Clays – silty clay to clay
      Zone 4: 2.60–2.95  Silt mixtures – clayey silt to silty clay
      Zone 5: 2.05–2.60  Sand mixtures – silty sand to sandy silt
      Zone 6: 1.31–2.05  Sands – clean sand to silty sand
      Zone 7: < 1.31  Gravelly sand to dense sand
    """
    if np.isnan(ic):
        return np.nan, np.nan
    if ic > 3.6:
        return 2, "Organic soils – clay"
    if 2.95 < ic <= 3.6:
        return 3, "Clays – silty clay to clay"
    if 2.60 < ic <= 2.95:
        return 4, "Silt mixtures – clayey silt to silty clay"
    if 2.05 < ic <= 2.60:
        return 5, "Sand mixtures – silty sand to sandy silt"
    if 1.31 < ic <= 2.05:
        return 6, "Sands – clean sand to silty sand"
    return 7, "Gravelly sand to dense sand"


def add_normalized_and_ic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      qt_kPa, qn_kPa
      Qt = (qt - sigma_v0) / sigma'_v0
      Fr = fs / (qt - sigma_v0) * 100
      Bq = (u2 - u0) / (qt - sigma_v0)
      Ic = sqrt[(3.47 - log10(Qt))^2 + (log10(Fr) + 1.22)^2]
      Zone, SoilBehaviorType
    """
    out = df.copy()

    out["qt_kPa"] = out["qt_MPa"] * 1000.0
    out["qn_kPa"] = out["qt_kPa"] - out["sigma_v0_kPa"]

    qn = out["qn_kPa"].to_numpy(dtype=float)
    sig_eff = out["sigma_v0_eff_kPa"].to_numpy(dtype=float)
    fs = out["fs_kPa"].to_numpy(dtype=float)
    u2 = out["u2_kPa"].to_numpy(dtype=float)
    u0 = out["u0_kPa"].to_numpy(dtype=float)

    qn_safe = np.where(qn <= 1e-9, np.nan, qn)
    sig_eff_safe = np.where(sig_eff <= 1e-9, np.nan, sig_eff)

    out["Qt"] = qn_safe / sig_eff_safe
    out["Fr_percent"] = (fs / qn_safe) * 100.0
    out["Bq"] = (u2 - u0) / qn_safe

    Qt_safe = out["Qt"].to_numpy(dtype=float)
    Qt_safe = np.where(Qt_safe <= 1e-12, np.nan, Qt_safe)
    Fr_safe = np.clip(out["Fr_percent"].to_numpy(dtype=float), 0.1, None)

    out["Ic"] = np.sqrt((3.47 - np.log10(Qt_safe))**2 + (np.log10(Fr_safe) + 1.22)**2)

    zones, sbt = [], []
    for ic in out["Ic"].to_numpy():
        z, t = ic_zone_and_sbt(ic)
        zones.append(z)
        sbt.append(t)

    out["Zone"] = zones
    out["SoilBehaviorType"] = sbt
    return out


# ----------------------------
# Derived parameters: cu, phi', Vs
# ----------------------------
def compute_cu_kPa(qt_MPa: np.ndarray, sigma_v0_kPa: np.ndarray, nkt: float) -> np.ndarray:
    """
    cu = (qt - sigma_v0) / Nkt
    Use consistent units: qt is MPa -> convert to kPa first, sigma_v0 is kPa.
    Returns cu in kPa.
    """
    qt_kPa = qt_MPa * 1000.0
    cu = (qt_kPa - sigma_v0_kPa) / nkt
    return np.where(cu < 0, np.nan, cu)


def compute_phi_prime_deg_from_Qt(Qt: np.ndarray) -> np.ndarray:
    """
    φ' = 11 * log10(Qt) + 17.6
    Returns degrees.
    """
    Qt_safe = np.where(Qt <= 1e-12, np.nan, Qt)
    return 11.0 * np.log10(Qt_safe) + 17.6


def compute_vs_mps(qt_MPa: np.ndarray, sigma_v0_eff_kPa: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """
    Vs depends on soil type (zone-based):
      - Sand (Zones 5–7): Baldi et al. 1986
          Vs = 277 * (qt)^0.13 * (σ'v0)^0.27, qt & σ'v0 in MPa
      - Clay/silt (Zones 2–4): Mayne & Rix 1995
          Vs = 1.75 * (qt)^0.627, qt in kPa
    Returns Vs in m/s.
    """
    zone = np.asarray(zone)
    vs = np.full_like(qt_MPa, np.nan, dtype=float)

    qt_MPa_safe = np.clip(np.asarray(qt_MPa, dtype=float), 1e-9, None)
    sig_eff_MPa = np.clip(np.asarray(sigma_v0_eff_kPa, dtype=float) / 1000.0, 1e-9, None)
    qt_kPa_safe = np.clip(qt_MPa_safe * 1000.0, 1e-9, None)

    sand_mask = np.isin(zone, [5, 6, 7])
    clay_mask = np.isin(zone, [2, 3, 4])

    vs[sand_mask] = 277.0 * (qt_MPa_safe[sand_mask] ** 0.13) * (sig_eff_MPa[sand_mask] ** 0.27)
    vs[clay_mask] = 1.75 * (qt_kPa_safe[clay_mask] ** 0.627)
    return vs


def add_strength_stiffness(out: pd.DataFrame) -> pd.DataFrame:
    out2 = out.copy()

    out2["cu_kPa"] = compute_cu_kPa(
        out2["qt_MPa"].to_numpy(dtype=float),
        out2["sigma_v0_kPa"].to_numpy(dtype=float),
        NKT
    )

    phi = compute_phi_prime_deg_from_Qt(out2["Qt"].to_numpy(dtype=float))
    zone_arr = out2["Zone"].to_numpy()
    # Typical: do not report phi' in clay/organic zones
    phi[np.isin(zone_arr, [2, 3])] = np.nan
    out2["phi_prime_deg"] = phi

    out2["Vs_mps"] = compute_vs_mps(
        out2["qt_MPa"].to_numpy(dtype=float),
        out2["sigma_v0_eff_kPa"].to_numpy(dtype=float),
        out2["Zone"].to_numpy()
    )

    return out2


# ----------------------------
# Dr for sands (your equation)
# ----------------------------
def compute_Dr_sand(
    qc_kPa: np.ndarray,
    sigma_v0_eff_kPa: np.ndarray,
    zone: np.ndarray,
    pa_kPa: float = PA_KPA,
    C0: float = DR_C0,
    C1: float = DR_C1,
    C2: float = DR_C2
) -> np.ndarray:
    """
    Dr = (1/C2) * ln( (qc/pa) / ( C0 * (σ'v0/pa)^C1 ) )
    Computed only for sand zones (5–7). Returns Dr array with NaN elsewhere.
    Dr is clipped to [0, 1].
    """
    Dr = np.full_like(qc_kPa, np.nan, dtype=float)
    sand_mask = np.isin(zone, [5, 6, 7])

    if not np.any(sand_mask):
        return Dr

    qc = np.clip(qc_kPa[sand_mask], 1e-6, None)
    sig = np.clip(sigma_v0_eff_kPa[sand_mask], 1e-6, None)

    arg = (qc / pa_kPa) / (C0 * (sig / pa_kPa) ** C1)
    arg = np.clip(arg, 1e-12, None)  # keep log valid

    Dr_s = (1.0 / C2) * np.log(arg)
    Dr[sand_mask] = np.clip(Dr_s, 0.0, 1.0)

    return Dr


# ----------------------------
# Stiffness Es (chosen with Dr(z) for sands)
# ----------------------------
def es_clay_gradual_kPa(qc_kPa: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Clay stiffness: Es = (3..8)*qc with gradual factor:
      - low qc -> factor close to 8
      - high qc -> factor close to 3
    """
    Es = np.full_like(qc_kPa, np.nan, dtype=float)
    if not np.any(mask):
        return Es

    qc_c = qc_kPa[mask]
    qc_min = np.nanmin(qc_c)
    qc_max = np.nanmax(qc_c)
    denom = (qc_max - qc_min) if (qc_max > qc_min) else 1.0
    t = (qc_c - qc_min) / denom  # 0 at low qc, 1 at high qc
    factor = 8.0 - 5.0 * np.clip(t, 0.0, 1.0)  # 8 -> 3
    Es[mask] = factor * qc_c
    return Es


def add_stiffness_Es(out: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      Dr_sand (-) (only zones 5–7)
      Es_chosen_kPa:
        - Zones 2–4: gradual clay/silt (3..8)qc
        - Zones 5–7: Es=(1+Dr^2)*qc using computed Dr(z)
    Also includes Es_sand_Dr_kPa as explicit sand formula output.
    """
    out2 = out.copy()

    qc_kPa = out2["qc_MPa"].to_numpy(dtype=float) * 1000.0
    zone = out2["Zone"].to_numpy()
    sig_eff_kPa = out2["sigma_v0_eff_kPa"].to_numpy(dtype=float)

    sand_mask = np.isin(zone, [5, 6, 7])
    clay_silt_mask = np.isin(zone, [2, 3, 4])

    # Dr for sands (your formula)
    out2["Dr_sand"] = compute_Dr_sand(qc_kPa, sig_eff_kPa, zone)

    # Clay/silt Es
    out2["Es_clay_gradual_kPa"] = es_clay_gradual_kPa(qc_kPa, clay_silt_mask)

    # Sand Es using Dr(z)
    Dr = out2["Dr_sand"].to_numpy(dtype=float)
    out2["Es_sand_Dr_kPa"] = np.where(sand_mask, (1.0 + Dr**2) * qc_kPa, np.nan)

    # Chosen Es
    Es_chosen = np.full_like(qc_kPa, np.nan, dtype=float)
    Es_chosen[clay_silt_mask] = out2["Es_clay_gradual_kPa"].to_numpy(dtype=float)[clay_silt_mask]
    Es_chosen[sand_mask] = out2["Es_sand_Dr_kPa"].to_numpy(dtype=float)[sand_mask]
    out2["Es_chosen_kPa"] = Es_chosen

    method = np.array([""] * len(out2), dtype=object)
    method[clay_silt_mask] = "Clay/Silt: gradual (3..8)qc"
    method[sand_mask] = "Sand: Es=(1+Dr(z)^2)*qc (Dr from ln-correlation)"
    out2["Es_method"] = method

    return out2


# ----------------------------
# PLOTTING
# ----------------------------
def add_layer_lines(ax, lines):
    for z in lines:
        ax.axhline(z, linewidth=1.5)
    return ax


def plot_stresses(out: pd.DataFrame):
    z = out["depth_m"].to_numpy()
    sv = out["sigma_v0_kPa"].to_numpy()
    u0 = out["u0_kPa"].to_numpy()
    sve = out["sigma_v0_eff_kPa"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

    axes[0].plot(sv, z)
    axes[0].set_title("σv0")
    axes[0].set_xlabel("kPa")

    axes[1].plot(u0, z)
    axes[1].set_title("u0")
    axes[1].set_xlabel("kPa")

    axes[2].plot(sve, z)
    axes[2].set_title("σ'v0")
    axes[2].set_xlabel("kPa")

    for ax in axes:
        ax.grid(True)
        ax.invert_yaxis()
        add_layer_lines(ax, LAYER_LINES_M)

    axes[0].set_ylabel("z [m]")
    fig.suptitle("Vertical stresses", y=0.98)
    fig.tight_layout()

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "01_stresses.png", dpi=200)
    plt.show()


def plot_cpt_panels(out: pd.DataFrame):
    z = out["depth_m"].to_numpy()

    qc = out["qc_MPa"].to_numpy()
    qt = out["qt_MPa"].to_numpy()
    fs_MPa = out["fs_kPa"].to_numpy() / 1000.0
    u2_MPa = out["u2_kPa"].to_numpy() / 1000.0
    Rf = out["Rf_percent"].to_numpy()
    gamma = out["gamma_kNm3"].to_numpy()

    fig, axes = plt.subplots(1, 6, figsize=(14, 6), sharey=True)

    axes[0].plot(qc, z); axes[0].set_title("qc [MPa]"); axes[0].set_xlabel("MPa")
    axes[1].plot(qt, z); axes[1].set_title("qt [MPa]"); axes[1].set_xlabel("MPa")
    axes[2].plot(fs_MPa, z); axes[2].set_title("fs [MPa]"); axes[2].set_xlabel("MPa")
    axes[3].plot(u2_MPa, z); axes[3].set_title("u2 [MPa]"); axes[3].set_xlabel("MPa")
    axes[4].plot(Rf, z); axes[4].set_title("Rf [%]"); axes[4].set_xlabel("%")
    axes[5].plot(gamma, z); axes[5].set_title("γ [kN/m³]"); axes[5].set_xlabel("kN/m³")

    for ax in axes:
        ax.grid(True)
        ax.invert_yaxis()
        add_layer_lines(ax, LAYER_LINES_M)

    axes[0].set_ylabel("z [m]")
    fig.suptitle("CPT parameters", y=0.98)
    fig.tight_layout()

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "02_cpt_panels.png", dpi=200)
    plt.show()


def plot_plasticity_index(out: pd.DataFrame):
    z = out["depth_m"].to_numpy()
    Ic = out["Ic"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))

    bands = [
        (1.00, 1.31, "Zone 7\nGravelly sand\nto dense sand"),
        (1.31, 2.05, "Zone 6\nSands\n(clean to silty)"),
        (2.05, 2.60, "Zone 5\nSand mixtures\n(silty sand–sandy silt)"),
        (2.60, 2.95, "Zone 4\nSilt mixtures\n(clayey silt–silty clay)"),
        (2.95, 3.60, "Zone 3\nClays\n(silty clay–clay)"),
        (3.60, 4.00, "Zone 2\nOrganic soils\n– clay"),
    ]
    band_colors = ["#a8d08d", "#9dc3e6", "#b4a7d6", "#f9cb9c", "#e6b8af", "#e06666"]

    for (x0, x1, label), c in zip(bands, band_colors):
        ax.axvspan(x0, x1, alpha=0.65, color=c, zorder=0)
        xm = 0.5 * (x0 + x1)
        ax.text(
            xm, 0.03, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=8, color="black"
        )

    ax.scatter(Ic, z, s=10, linewidths=0, zorder=2)

    ax.set_xlim(1.0, 4.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("z [m]")
    ax.set_title("Plasticity index (Ic) as a function of depth")

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Ic [-]")

    for b in [1.31, 2.05, 2.60, 2.95, 3.60]:
        ax.axvline(b, linewidth=1.0, alpha=0.5)

    add_layer_lines(ax, LAYER_LINES_M)

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "03_plasticity_index_Ic.png", dpi=200)

    plt.tight_layout()
    plt.show()


def plot_derived_panels(out: pd.DataFrame):
    z = out["depth_m"].to_numpy()
    cu = out["cu_kPa"].to_numpy()
    phi = out["phi_prime_deg"].to_numpy()
    vs = out["Vs_mps"].to_numpy()

    # Es stored kPa -> plot MPa
    Es_MPa = out["Es_chosen_kPa"].to_numpy() / 1000.0

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 6), sharey=True)

    axes[0].plot(cu, z);      axes[0].set_title("cu (Cu)");      axes[0].set_xlabel("kPa")
    axes[1].plot(phi, z);     axes[1].set_title("φ′");           axes[1].set_xlabel("deg")
    axes[2].plot(vs, z);      axes[2].set_title("Vs");           axes[2].set_xlabel("m/s")
    axes[3].plot(Es_MPa, z);  axes[3].set_title("Es (chosen)");  axes[3].set_xlabel("MPa")

    for ax in axes:
        ax.grid(True)
        ax.invert_yaxis()
        add_layer_lines(ax, LAYER_LINES_M)

    axes[0].set_ylabel("z [m]")
    fig.suptitle("Derived parameters (cu, φ′, Vs, Es)", y=0.98)
    fig.tight_layout()

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "04_derived_panels.png", dpi=200)
    plt.show()


def plot_es_options(out: pd.DataFrame):
    """
    Shows Es vs depth for clay/silt (gradual) and sand (chosen Dr-based formula).
    NOTE: Es stored kPa, plotted MPa.
    """
    z = out["depth_m"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Clay/silt
    axes[0].plot(out["Es_clay_gradual_kPa"].to_numpy() / 1000.0, z)
    axes[0].set_title("Clay/Silt Es options")
    axes[0].set_xlabel("MPa")
    axes[0].text(0.02, 0.95, "Shown: gradual (3..8)qc", transform=axes[0].transAxes, va="top")

    # Sand (chosen)
    axes[1].plot(out["Es_sand_Dr_kPa"].to_numpy() / 1000.0, z, label="Chosen: (1+Dr(z)^2)*qc")
    axes[1].set_title("Sand Es (chosen)")
    axes[1].set_xlabel("MPa")
    axes[1].legend(loc="best", fontsize=9)

    for ax in axes:
        ax.grid(True)
        ax.invert_yaxis()
        add_layer_lines(ax, LAYER_LINES_M)

    axes[0].set_ylabel("z [m]")
    fig.suptitle("Stiffness (Es) from CPT correlations", y=0.98)
    fig.tight_layout()

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "05_Es_options.png", dpi=200)
    plt.show()


# ----------------------------
# EXCEL REPORT
# ----------------------------
def autosize_columns(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                v = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(v))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 45)


def write_clean_excel(summary: dict, out: pd.DataFrame, out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(list(summary.items()), columns=["Item", "Value"])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        cols = [
            "depth_m",
            "qc_MPa", "qt_MPa",
            "fs_kPa", "u2_kPa", "a",
            "Rf_percent",
            "gamma_kNm3",
            "sigma_v0_kPa", "u0_kPa", "sigma_v0_eff_kPa",
            "qt_kPa", "qn_kPa",
            "Qt", "Fr_percent", "Bq",
            "Ic", "Zone", "SoilBehaviorType",
            "cu_kPa", "phi_prime_deg", "Vs_mps",
            "Dr_sand",
            "Es_chosen_kPa", "Es_method",
            "Es_clay_gradual_kPa",
            "Es_sand_Dr_kPa",
        ]
        cols = [c for c in cols if c in out.columns]
        out[cols].to_excel(writer, sheet_name="Results", index=False)

        wb = writer.book
        ws_sum = wb["Summary"]
        ws_res = wb["Results"]

        ws_res.freeze_panes = "A2"
        ws_sum.freeze_panes = "A2"

        autosize_columns(ws_sum)
        autosize_columns(ws_res)

    print(f"Saved Excel report: {out_path.resolve()}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    print("Script folder:", BASE_DIR)
    print("Reading Excel:", EXCEL_PATH.resolve())

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH.resolve()}")

    df = read_cpt_excel(EXCEL_PATH, SHEET_NAME)

    # Basic CPT
    df["qt_MPa"] = compute_qt_MPa(df["qc_MPa"].values, df["u2_kPa"].values, df["a"].values)
    df["Rf_percent"] = compute_Rf_percent(df["fs_kPa"].values, df["qt_MPa"].values)

    # Unit weight
    df["gamma_kNm3"] = estimate_gamma_kNm3(df["Rf_percent"].values, df["qt_MPa"].values)

    # Stresses
    z, sv, u0, sve = compute_stresses(df["depth_m"].values, df["gamma_kNm3"].values)
    stress_df = pd.DataFrame({
        "depth_m": z,
        "sigma_v0_kPa": sv,
        "u0_kPa": u0,
        "sigma_v0_eff_kPa": sve
    })

    # Merge (outer join to include seabed row z=0)
    out = pd.merge(stress_df, df, on="depth_m", how="left").sort_values("depth_m").reset_index(drop=True)

    # Normalized + Ic
    out = add_normalized_and_ic(out)

    # Add cu, phi', Vs
    out = add_strength_stiffness(out)

    # Add Dr and Es (chosen)
    out = add_stiffness_Es(out)

    # ----------------------------
    # AVERAGE Dr (Relative density) in requested layers
    # NOTE: Dr_sand is only defined for sand zones (5–7); elsewhere it is NaN.
    # ----------------------------
    DR_LAYERS = [
        (3.3, 13.6),
        (13.6, 25.0),
    ]

    print("\nAverage Dr_sand in layers:")
    print("Layer [m]      Dr_mean [-]   n_sand_points")

    for z_top, z_bot in DR_LAYERS:
        mask = (out["depth_m"] >= z_top) & (out["depth_m"] < z_bot)

        Dr_vals = out.loc[mask, "Dr_sand"].to_numpy(dtype=float)
        Dr_mean = np.nanmean(Dr_vals)
        n_sand = int(np.sum(~np.isnan(Dr_vals)))

        print(f"{z_top:4.1f}–{z_bot:<4.1f}     {Dr_mean:10.3f}     {n_sand:12d}")


    # ----------------------------
    # LAYER AVERAGES (phi, cu, Es)
    # ----------------------------
    LAYERS = [
        (3.3, 13.6),
        (13.6, 25.0),
        (25.0, 28.7),
        (28.7, 36.0),
        (36.0, 41.2),
    ]

    print("\nLayer-averaged parameters:")
    print("Layer [m]      phi' [deg]    cu [kPa]     Es [MPa]")
    print("-" * 55)

    for z_top, z_bot in LAYERS:
        mask = (out["depth_m"] >= z_top) & (out["depth_m"] < z_bot)

        phi_mean = np.nanmean(out.loc[mask, "phi_prime_deg"])
        cu_mean = np.nanmean(out.loc[mask, "cu_kPa"])
        Es_mean = np.nanmean(out.loc[mask, "Es_chosen_kPa"]) / 1000.0  # kPa → MPa

        print(
            f"{z_top:4.1f}–{z_bot:<4.1f}     "
            f"{phi_mean:8.2f}     "
            f"{cu_mean:8.1f}     "
            f"{Es_mean:8.2f}"
        )


    # Summary
    summary = {
        "Water depth zw (m)": ZW_WATER_DEPTH_M,
        "Gamma_w (kN/m³)": GAMMA_W,
        "Pa (kPa)": PA_KPA,
        "Nkt (-)": NKT,
        "Dr constants (C0,C1,C2)": f"{DR_C0}, {DR_C1}, {DR_C2}",
        "Rows in input": len(df),
        "Max depth (m)": float(np.nanmax(out["depth_m"].to_numpy())),
        "qt max (MPa)": float(np.nanmax(out["qt_MPa"].to_numpy())),
        "Rf max (%)": float(np.nanmax(out["Rf_percent"].to_numpy())),
        "Ic min (-)": float(np.nanmin(out["Ic"].to_numpy())),
        "Ic max (-)": float(np.nanmax(out["Ic"].to_numpy())),
        "cu max (kPa)": float(np.nanmax(out["cu_kPa"].to_numpy())),
        "phi' max (deg)": float(np.nanmax(out["phi_prime_deg"].to_numpy())),
        "Vs max (m/s)": float(np.nanmax(out["Vs_mps"].to_numpy())),
        "Dr max (-)": float(np.nanmax(out["Dr_sand"].to_numpy())),
        "Es max (kPa)": float(np.nanmax(out["Es_chosen_kPa"].to_numpy())),
    }

    

    # Write Excel
    write_clean_excel(summary, out, OUT_EXCEL)

    # Plots
    plot_stresses(out)
    plot_cpt_panels(out)
    plot_plasticity_index(out)
    plot_derived_panels(out)
    plot_es_options(out)

    # Terminal preview
    print("\nPreview (first 12 rows):")
    preview_cols = [
        "depth_m", "qc_MPa", "qt_MPa", "Rf_percent", "gamma_kNm3",
        "sigma_v0_kPa", "u0_kPa", "sigma_v0_eff_kPa",
        "Qt", "Fr_percent", "Ic", "Zone",
        "cu_kPa", "phi_prime_deg", "Vs_mps",
        "Dr_sand", "Es_chosen_kPa", "Es_method"
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out[preview_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()


