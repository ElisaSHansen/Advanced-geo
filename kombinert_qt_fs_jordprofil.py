import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ----------------------------
# INNSTILLINGER
# ----------------------------
CLASSIFIED_CSV = "cpt_profile_1_with_robertson1986.csv"

# Laggrenser (m)
layers = [
    (0.0, 3.3),
    (3.3, 13.6),
    (13.6, 25),
    (25, 28.7),
    (28.7, 36.0),
]

layer_bounds = sorted(set([b for pair in layers for b in pair]))

ZONE_COLORS = {
    1: "#4C72B0",
    2: "#00FF26",
    3: "#C44E52",
    4: "#342563",
    5: "#DC69F6",
    6: "#EEE758",
    7: "#46A333",
    8: "#FD5252",
    9: "#FD5252",
    10: "#FF7F0E",
    11: "#A6761D",
    12: "#84FFFF",
    0: "#BDBDBD",
}

ZONE_NAMES = {
    1: "1 = Sensitive clay",
    2: "2 = Organic soil",
    3: "3 = Clay",
    4: "4 = Silty clay",
    5: "5 = Clayey silt",
    6: "6 = Sandy silt",
    7: "7 = Silty sand",
    8: "8 = Sand to silty sand",
    9: "9 = Sand",
    10: "10 = Gravelly sand",
}

# ----------------------------
# LES INN DATA
# ----------------------------
df = pd.read_csv(CLASSIFIED_CSV, sep=";", decimal=",")
df.columns = df.columns.str.strip()

if len(df.columns) == 1 and "," in df.columns[0]:
    df = pd.read_csv(CLASSIFIED_CSV, sep=",")
    df.columns = df.columns.str.strip()

required = ["Depth_m", "qt_MPa", "fs_kPa", "u2_kPa", "robertson_zone"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Mangler kolonnen: {c}. Fant: {list(df.columns)}")

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required).sort_values("Depth_m").copy()

depth = df["Depth_m"].to_numpy(dtype=float)
qt = df["qt_MPa"].to_numpy(dtype=float)
fs = df["fs_kPa"].to_numpy(dtype=float)
u2 = df["u2_kPa"].to_numpy(dtype=float)
zone = df["robertson_zone"].to_numpy(dtype=int)

zmin = 0.0
zmax = float(np.nanmax(depth))

# ----------------------------
# PORE PRESSURE BEREGNING
# ----------------------------
GW_LEVEL = 0.0      # Juster hvis nødvendig
gamma_w = 9.81      # kN/m³
gamma_soil = 18.0   # kN/m³ (antatt konstant)

# Hydrostatisk poretrykk
u0 = np.where(depth > GW_LEVEL,
              gamma_w * (depth - GW_LEVEL),
              0.0)

# Excess pore pressure
delta_u = u2 - u0

# Totalspenning
sigma_v = gamma_soil * depth

# qt i kPa
qt_kPa = qt * 1000.0

# Pore pressure ratio Bq
Bq = np.full_like(qt_kPa, np.nan)
mask = (qt_kPa - sigma_v) > 0
Bq[mask] = delta_u[mask] / (qt_kPa[mask] - sigma_v[mask])

# ----------------------------
# LAG-STATISTIKK
# ----------------------------
layer_stats = []
for z0, z1 in layers:
    m = (depth >= z0) & (depth < z1)
    n = int(m.sum())

    qt_mean = float(np.nanmean(qt[m])) if n > 0 else np.nan
    fs_mean = float(np.nanmean(fs[m])) if n > 0 else np.nan
    u2_mean = float(np.nanmean(u2[m])) if n > 0 else np.nan
    Bq_mean = float(np.nanmean(Bq[m])) if n > 0 else np.nan

    layer_stats.append((z0, z1, n, qt_mean, fs_mean, u2_mean, Bq_mean))

print("\nAverage per layer:")
print("Layer (m)\tN\tqt (MPa)\tfs (kPa)\tu2 (kPa)\tBq")
for z0, z1, n, qt_mean, fs_mean, u2_mean, Bq_mean in layer_stats:
    print(f"{z0:>4.1f}-{z1:<4.1f}\t{n:>6d}\t"
          f"{qt_mean:>8.3f}\t{fs_mean:>8.2f}\t"
          f"{u2_mean:>8.2f}\t{Bq_mean:>8.3f}")

# ----------------------------
# HJELPEFUNKSJON
# ----------------------------
def compress_intervals(depth_arr, zone_arr):
    order = np.argsort(depth_arr)
    d = depth_arr[order]
    z = zone_arr[order]

    if len(d) == 0:
        return []

    if len(d) > 1:
        dz = np.diff(d)
        dz = dz[np.isfinite(dz) & (dz > 0)]
        half = 0.5 * (np.median(dz) if len(dz) else 0.05)
    else:
        half = 0.05

    intervals = []
    start = d[0] - half
    cur_zone = z[0]
    last_center = d[0]

    for i in range(1, len(d)):
        if z[i] != cur_zone:
            end = last_center + half
            intervals.append((start, end, cur_zone))
            start = d[i] - half
            cur_zone = z[i]
        last_center = d[i]

    end = last_center + half
    intervals.append((start, end, cur_zone))
    return intervals

intervals = compress_intervals(depth, zone)

# ----------------------------
# PLOT: qt | fs | u2 | soil profile
# ----------------------------
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(13, 7))
ax_qt, ax_fs, ax_u2, ax_prof = axes

# qt
ax_qt.plot(qt, depth, linewidth=1.5)
ax_qt.set_xlabel("qt (MPa)")
ax_qt.set_ylabel("Depth (m)")
ax_qt.grid(True, color="0.75")

# fs
ax_fs.plot(fs, depth, linewidth=1.5)
ax_fs.set_xlabel("fs (kPa)")
ax_fs.grid(True, color="0.75")

# u2
ax_u2.plot(u2, depth, linewidth=1.5)
ax_u2.set_xlabel("u2 (kPa)")
ax_u2.grid(True, color="0.75")

# soil profile
for z0, z1, zid in intervals:
    ax_prof.fill_between([0, 1], [z0, z0], [z1, z1],
                         color=ZONE_COLORS.get(int(zid), ZONE_COLORS[0]),
                         linewidth=0)

ax_prof.set_xlim(0, 1)
ax_prof.set_xticks([])
ax_prof.set_xlabel("Soil profile")

# Layer means
for (z0, z1, n, qt_mean, fs_mean, u2_mean, Bq_mean) in layer_stats:
    if np.isfinite(qt_mean):
        ax_qt.vlines(qt_mean, z0, z1, linestyles="--", linewidth=1.2, color="0.2")
    if np.isfinite(fs_mean):
        ax_fs.vlines(fs_mean, z0, z1, linestyles="--", linewidth=1.2, color="0.2")
    if np.isfinite(u2_mean):
        ax_u2.vlines(u2_mean, z0, z1, linestyles="--", linewidth=1.2, color="0.2")

# Røde laggrenser
for b in layer_bounds:
    for ax in axes:
        ax.axhline(b, color="red", linewidth=1.2)

for ax in axes:
    ax.set_ylim(zmax, zmin)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

from matplotlib.patches import Patch

legend_elements = []

for zid in sorted(ZONE_NAMES.keys()):
    legend_elements.append(
        Patch(facecolor=ZONE_COLORS.get(zid, "#BDBDBD"),
              label=ZONE_NAMES[zid])
    )

ax_prof.legend(handles=legend_elements,
               loc="center left",
               bbox_to_anchor=(1.05, 0.5),
               fontsize=8,
               frameon=True)

plt.tight_layout()
plt.show()