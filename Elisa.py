import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# INPUT
# ----------------------------
CSV_FILE = "cpt_profile_1.csv"
zw = 22.0          # m water depth
gamma_w = 9.81     # kN/m3
pa = 100.0         # kPa
Nkt = 17.0         # given in assignment

# ENKEL antakelse for effektivspenning:
# sigma'_v0(z) = integral gamma' dz ≈ gamma' * z
# (Bytt gamma_sub hvis dere bestemmer enhetsvekt per lag/jordtype)
gamma_sub = 9.0    # kN/m3 (typisk), dvs kPa/m

# ----------------------------
# LES DATA
# ----------------------------
df = pd.read_csv(CSV_FILE, sep=";", decimal=",")
cols = ["Depth_m", "qc_MPa", "fs_kPa", "u2_kPa", "alpha"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols).sort_values("Depth_m").reset_index(drop=True)

z = df["Depth_m"].to_numpy()

# ----------------------------
# 1) qt (MPa)
# qt = qc + (1 - alpha) * u2   (u2 kPa -> MPa)
# ----------------------------
df["qt_MPa"] = df["qc_MPa"] + (1.0 - df["alpha"]) * (df["u2_kPa"] / 1000.0)

qt_kPa = df["qt_MPa"].to_numpy() * 1000.0
fs_kPa = df["fs_kPa"].to_numpy()

# ----------------------------
# 2) Rf = 100 * fs/qt
# ----------------------------
df["Rf_pct"] = 100.0 * fs_kPa / np.maximum(qt_kPa, 1.0)

# ----------------------------
# 3) Offshore stresses
# u0 = gamma_w*(zw+z)
# sigma'v0 ≈ gamma' * z
# sigma_v0 = u0 + sigma'v0
# ----------------------------
u0 = gamma_w * (zw + z)              # kPa
sigma_v0_eff = np.maximum(gamma_sub * z, 1.0)   # kPa
sigma_v0 = u0 + sigma_v0_eff         # kPa

df["u0_kPa"] = u0
df["sigma_v0_eff_kPa"] = sigma_v0_eff
df["sigma_v0_kPa"] = sigma_v0

# ----------------------------
# 4) Undrained shear strength su
# qn = qt - sigma_v0
# su = qn/Nkt
# ----------------------------
qn = np.maximum(qt_kPa - sigma_v0, 1.0)
df["qn_kPa"] = qn
df["su_kPa"] = qn / Nkt

# ----------------------------
# 5) Relative density Dr (Jamiolkowski et al. 2001)
# Dr = (1/C2)*ln[ (qc/pa) / (C0*(sigma'/pa)^C1 ) ]
# constants from slide
# ----------------------------
C0, C1, C2 = 17.68, 0.5, 3.1
qc_kPa = df["qc_MPa"].to_numpy() * 1000.0

Dr = (1.0 / C2) * np.log(
    (np.maximum(qc_kPa / pa, 1e-6)) /
    (C0 * (np.maximum(sigma_v0_eff / pa, 1e-6) ** C1))
)
df["Dr"] = Dr  # dette er "korrelasjons-Dr" (ikke klippet til 0-1)

# ----------------------------
# 6) phi' (Kulhawy & Mayne, bruker Qt = qn/sigma')
# phi' = 11 log10(Qt) + 17.6
# ----------------------------
Qt = qn / sigma_v0_eff
df["Qt"] = Qt
df["phi_deg"] = 11.0 * np.log10(np.maximum(Qt, 1e-6)) + 17.6

# ----------------------------
# 7) Eoed = alpha_oed * qc  (velg alpha_oed)
# Her setter vi en enkel konstant (du kan gjøre den lagvis basert på jordtype senere)
# ----------------------------
alpha_oed = 3.0
df["Eoed_MPa"] = alpha_oed * df["qc_MPa"]  # MPa hvis qc i MPa

# ----------------------------
# 8) Vs (to uttrykk – velg basert på jordtype senere)
# Sand (Baldi): Vs = 277*(qt)^0.13*(sigma')^0.27  (qt og sigma' i MPa)
# Clay (Mayne&Rix): Vs = 1.75*(qt_kPa)^0.627
# ----------------------------
sigma_eff_MPa = sigma_v0_eff / 1000.0
Vs_sand = 277.0 * (np.maximum(df["qt_MPa"].to_numpy(), 1e-6) ** 0.13) * (np.maximum(sigma_eff_MPa, 1e-6) ** 0.27)
Vs_clay = 1.75 * (np.maximum(qt_kPa, 1e-6) ** 0.627)

df["Vs_sand_ms"] = Vs_sand
df["Vs_clay_ms"] = Vs_clay

# ----------------------------
# PLOTT (langs dybde) – slik assignment ber om
# ----------------------------
fig, axes = plt.subplots(1, 6, sharey=True, figsize=(16, 7))

axes[0].plot(df["qt_MPa"], z);        axes[0].set_xlabel("qt (MPa)"); axes[0].grid(True)
axes[1].plot(df["Rf_pct"], z);        axes[1].set_xlabel("Rf (%)");   axes[1].grid(True)
axes[2].plot(df["su_kPa"], z);        axes[2].set_xlabel("su (kPa)"); axes[2].grid(True)
axes[3].plot(df["phi_deg"], z);       axes[3].set_xlabel("phi' (deg)"); axes[3].grid(True)
axes[4].plot(df["Eoed_MPa"], z);      axes[4].set_xlabel("Eoed (MPa)"); axes[4].grid(True)
axes[5].plot(df["Vs_sand_ms"], z, label="Vs sand")
axes[5].plot(df["Vs_clay_ms"], z, label="Vs clay")
axes[5].set_xlabel("Vs (m/s)")
axes[5].grid(True)
axes[5].legend()

for ax in axes:
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

axes[0].set_ylabel("Depth (m)")
axes[0].invert_yaxis()

plt.tight_layout()
plt.show()

df.to_csv("cpt_profile_1_part1_results.csv", index=False, sep=";", decimal=",")
print("Skrev: cpt_profile_1_part1_results.csv")
