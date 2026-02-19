import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CPT REPORT SCRIPT (Assignment 1 - Part 1)
# - Computes: qt, Rf, gamma, stresses, qn, Qt, Fr, Bq, Ic + soil zones
# - Exports: clean Excel report
# - Plots: stresses, CPT panels, plasticity index chart with zones
# ============================================================

# ----------------------------
# USER SETTINGS
# ----------------------------
BASE_DIR = Path(__file__).parent
EXCEL_PATH = BASE_DIR / "cpt profile 1 (1).xlsx"
SHEET_NAME = "Data Sheet"

ZW_WATER_DEPTH_M = 22.0   # water depth (m)
GAMMA_W = 9.81            # kN/m^3
PA_KPA = 100.0            # kPa (atmospheric pressure)
PA_MPA = PA_KPA / 1000.0  # MPa

OUT_EXCEL = BASE_DIR / "Part1_CPT_Report.xlsx"

# Optional: horizontal red layer lines (edit to match your chosen layer boundaries)
LAYER_LINES_M = []  # set [] if you don't want them

# Save figures as PNG too
SAVE_FIGS = True


# ----------------------------
# HELPERS: Reading
# ----------------------------
def read_cpt_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Reads the CPT excel format where data start after some header rows.
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

    # Fill missing u2/a if absent
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
# HELPERS: CPT calculations
# ----------------------------
def compute_qt_MPa(qc_MPa: np.ndarray, u2_kPa: np.ndarray, a: np.ndarray) -> np.ndarray:
    # qt = qc + (1-a)u2 with consistent units (u2: kPa -> MPa)
    return qc_MPa + (1.0 - a) * (u2_kPa / 1000.0)


def compute_Rf_percent(fs_kPa: np.ndarray, qt_MPa: np.ndarray) -> np.ndarray:
    # Rf(%) = fs / qt *100 ; convert qt MPa -> kPa
    qt_kPa = qt_MPa * 1000.0
    qt_kPa = np.where(qt_kPa <= 1e-9, np.nan, qt_kPa)
    return (fs_kPa / qt_kPa) * 100.0


def estimate_gamma_kNm3(Rf_percent: np.ndarray, qt_MPa: np.ndarray) -> np.ndarray:
    """
    Uses your shown correlation:
      gamma/gamma_w = 0.27 log10(Rf) + 0.36 log10(qt/pa) + 1.236
    Rf in %, qt & pa in MPa, gamma_w in kN/m^3
    """
    Rf_safe = np.clip(Rf_percent, 0.1, None)      # avoid log(0)
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


def ic_zone_and_sbt(ic: float):
    """
    Based on your table (Ic ranges):
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
      Zone, SoilBehaviorType (from Ic table)
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

    zones = []
    sbt = []
    for ic in out["Ic"].to_numpy():
        z, t = ic_zone_and_sbt(ic)
        zones.append(z)
        sbt.append(t)
    out["Zone"] = zones
    out["SoilBehaviorType"] = sbt

    return out


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
    fs_MPa = (out["fs_kPa"].to_numpy()) / 1000.0
    u2_MPa = (out["u2_kPa"].to_numpy()) / 1000.0
    Rf = out["Rf_percent"].to_numpy()
    gamma = out["gamma_kNm3"].to_numpy()

    fig, axes = plt.subplots(1, 6, figsize=(14, 6), sharey=True)

    axes[0].plot(qc, z)
    axes[0].set_title("qc [MPa]")
    axes[0].set_xlabel("MPa")

    axes[1].plot(qt, z)
    axes[1].set_title("qt [MPa]")
    axes[1].set_xlabel("MPa")

    axes[2].plot(fs_MPa, z)
    axes[2].set_title("fs [MPa]")
    axes[2].set_xlabel("MPa")

    axes[3].plot(u2_MPa, z)
    axes[3].set_title("u2 [MPa]")
    axes[3].set_xlabel("MPa")

    axes[4].plot(Rf, z)
    axes[4].set_title("Rf [%]")
    axes[4].set_xlabel("%")

    axes[5].plot(gamma, z)
    axes[5].set_title("γ [kN/m³]")
    axes[5].set_xlabel("kN/m³")

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

    # --- Zone bands and labels (based on your table) ---
    # Zone 7: Ic < 1.31
    # Zone 6: 1.31–2.05
    # Zone 5: 2.05–2.60
    # Zone 4: 2.60–2.95
    # Zone 3: 2.95–3.60
    # Zone 2: > 3.60
    bands = [
        (1.00, 1.31, "Zone 7\nGravelly sand\nto dense sand"),
        (1.31, 2.05, "Zone 6\nSands\n(clean to silty)"),
        (2.05, 2.60, "Zone 5\nSand mixtures\n(silty sand–sandy silt)"),
        (2.60, 2.95, "Zone 4\nSilt mixtures\n(clayey silt–silty clay)"),
        (2.95, 3.60, "Zone 3\nClays\n(silty clay–clay)"),
        (3.60, 4.00, "Zone 2\nOrganic soils\n– clay"),
    ]

    # Choose band colors similar to your example
    # (Matplotlib defaults are fine too, but here we match the look)
    band_colors = [
        "#a8d08d",  # green
        "#9dc3e6",  # light blue
        "#b4a7d6",  # purple
        "#f9cb9c",  # orange
        "#e6b8af",  # pink
        "#e06666",  # red
    ]

    for (x0, x1, label), c in zip(bands, band_colors):
        ax.axvspan(x0, x1, alpha=0.65, color=c, zorder=0)

        # Put zone text near the top of plot area
        xm = 0.5 * (x0 + x1)
        ax.text(
            xm, 0.03, label,
            transform=ax.get_xaxis_transform(),  # y in axes fraction
            ha="center", va="bottom",
            fontsize=8, color="black"
        )

    # Plot points
    ax.scatter(Ic, z, s=10, linewidths=0, zorder=2)

    # Axes style
    ax.set_xlim(1.0, 4.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("z [m]")
    ax.set_title("Plasticity index (Ic) as a function of depth")

    # Put x-axis ticks on top like your figure
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Ic [-]")

    # Optional: boundary lines at zone transitions
    boundaries = [1.31, 2.05, 2.60, 2.95, 3.60]
    for b in boundaries:
        ax.axvline(b, linewidth=1.0, alpha=0.5)

    # Optional: your horizontal layer lines (remove if you don't want them)
    add_layer_lines(ax, LAYER_LINES_M)

    if SAVE_FIGS:
        fig.savefig(BASE_DIR / "03_plasticity_index_Ic.png", dpi=200)

    plt.tight_layout()
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
        ws.column_dimensions[col_letter].width = min(max_len + 2, 35)


def write_clean_excel(summary: dict, out: pd.DataFrame, out_path: Path):
    """
    Writes:
      - Summary sheet (inputs + key stats)
      - Results sheet (clean columns + freeze header)
    """
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(list(summary.items()), columns=["Item", "Value"])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Cleaner column order
        cols = [
            "depth_m",
            "qc_MPa", "qt_MPa",
            "fs_kPa", "u2_kPa", "a",
            "Rf_percent",
            "gamma_kNm3",
            "sigma_v0_kPa", "u0_kPa", "sigma_v0_eff_kPa",
            "qn_kPa", "Qt", "Fr_percent", "Bq",
            "Ic", "Zone", "SoilBehaviorType"
        ]
        cols = [c for c in cols if c in out.columns]
        out[cols].to_excel(writer, sheet_name="Results", index=False)

        wb = writer.book
        ws_sum = wb["Summary"]
        ws_res = wb["Results"]

        # Freeze header row
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

    # Unit weight (your formula)
    df["gamma_kNm3"] = estimate_gamma_kNm3(df["Rf_percent"].values, df["qt_MPa"].values)

    # Stresses (kPa)
    z, sv, u0, sve = compute_stresses(df["depth_m"].values, df["gamma_kNm3"].values)
    stress_df = pd.DataFrame({
        "depth_m": z,
        "sigma_v0_kPa": sv,
        "u0_kPa": u0,
        "sigma_v0_eff_kPa": sve
    })

    # Merge (outer join for seabed row)
    out = pd.merge(stress_df, df, on="depth_m", how="left").sort_values("depth_m").reset_index(drop=True)

    # Normalized + Ic
    out = add_normalized_and_ic(out)

    # Summary stats
    summary = {
        "Water depth zw (m)": ZW_WATER_DEPTH_M,
        "Gamma_w (kN/m³)": GAMMA_W,
        "Pa (kPa)": PA_KPA,
        "Rows in input": len(df),
        "Max depth (m)": float(np.nanmax(out["depth_m"].to_numpy())),
        "qt max (MPa)": float(np.nanmax(out["qt_MPa"].to_numpy())),
        "Rf max (%)": float(np.nanmax(out["Rf_percent"].to_numpy())),
        "Ic min (-)": float(np.nanmin(out["Ic"].to_numpy())),
        "Ic max (-)": float(np.nanmax(out["Ic"].to_numpy()))
    }

    # Write Excel report
    write_clean_excel(summary, out, OUT_EXCEL)

    # PLOTS (order you asked: stresses first)
    plot_stresses(out)
    plot_cpt_panels(out)
    plot_plasticity_index(out)

    # Terminal preview (clean)
    print("\nPreview (first 12 rows):")
    preview_cols = ["depth_m", "qc_MPa", "qt_MPa", "Rf_percent", "gamma_kNm3",
                    "sigma_v0_kPa", "u0_kPa", "sigma_v0_eff_kPa", "Qt", "Fr_percent", "Ic", "SoilBehaviorType"]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out[preview_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
