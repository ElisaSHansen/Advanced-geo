import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ----------------------------
# FILNAVN
# ----------------------------
CSV_PATH = "cpt_profile_1_korrigert.csv"

# ----------------------------
# LES INN DATA
# ----------------------------
df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
df.columns = df.columns.str.strip()

cols = ["Depth_m", "qt_MPa", "fs_kPa", "u2_kPa", "FR_percent"]
for c in cols:
    if c not in df.columns:
        raise ValueError(f"Mangler kolonnen: {c}. Fant: {list(df.columns)}")
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=cols).sort_values("Depth_m").copy()

depth = df["Depth_m"].to_numpy(dtype=float)
qt = df["qt_MPa"].to_numpy(dtype=float)
fs = df["fs_kPa"].to_numpy(dtype=float)
u2 = df["u2_kPa"].to_numpy(dtype=float)
Rf = df["FR_percent"].to_numpy(dtype=float)

# ----------------------------
# BEREGN gamma'
# gamma'/gamma_w = 0.27*log10(Rf) + 0.36*log10(qt/pa) + 1.236
# ----------------------------
pa_MPa = 0.1
gamma_w = 9.81  # kN/m3

gamma_ratio = np.full_like(qt, np.nan, dtype=float)
mask = (Rf > 0) & (qt > 0)

gamma_ratio[mask] = (
    0.27 * np.log10(Rf[mask]) +
    0.36 * np.log10(qt[mask] / pa_MPa) +
    1.236
)

gamma_eff = gamma_ratio * gamma_w  # γ'

# ----------------------------
# AKSER
# ----------------------------
ZMIN = 0.0
ZMAX = 35.0
FR_LIM = 15.0

# ----------------------------
# PLOT
# ----------------------------
fig, axes = plt.subplots(1, 5, sharey=True, figsize=(18, 7))
ax_qt, ax_fs, ax_u2, ax_fr, ax_g = axes

grid_color = "0.75"

# qt
ax_qt.plot(qt, depth, color="red", linewidth=1.6)
ax_qt.set_xlabel("qt (MPa)")
ax_qt.set_ylabel("Depth (m)")
ax_qt.set_xlim(left=0)
ax_qt.grid(True, color=grid_color)

# fs
ax_fs.plot(fs, depth, color="green", linewidth=1.6)
ax_fs.set_xlabel("fs (kPa)")
ax_fs.set_xlim(left=0)
ax_fs.grid(True, color=grid_color)

# u2
ax_u2.plot(u2, depth, color="blue", linewidth=1.6)
ax_u2.set_xlabel("u2 (kPa)")
ax_u2.set_xlim(-500, 2500)
ax_u2.set_xticks([-500, 0, 500, 1000, 1500, 2000, 2500])
ax_u2.grid(True, color=grid_color)

# Rf
ax_fr.plot(Rf, depth, color="orange", linewidth=1.6)
ax_fr.set_xlabel("Rf (%)")
ax_fr.set_xlim(0, FR_LIM)
ax_fr.set_xticks([0, 5, 10, 15])
ax_fr.grid(True, color=grid_color)

# gamma'
ax_g.plot(gamma_eff, depth, color="black", linewidth=1.6)
ax_g.set_xlabel("γ (kN/m³)")
ax_g.set_xlim(10, 22)
ax_g.grid(True, color=grid_color)

# ----------------------------
# FELLES OPPSETT
# ----------------------------
for ax in axes:
    ax.set_ylim(ZMAX, ZMIN)
    ax.set_yticks(np.arange(0, 36, 5))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.grid(which="major", axis="y", color=grid_color)

plt.tight_layout()
plt.show()
