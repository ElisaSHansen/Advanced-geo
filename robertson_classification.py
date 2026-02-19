import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ----------------------------
# INNSTILLINGER
# ----------------------------
CSV_PATH = "cpt_profile_1_korrigert.csv"       # din fil med qt_MPa og FR_percent
ZONES_JSON = "zones_robertson1986.json"        # polygonene
ROBERTSON_IMAGE = "robertson1986.png"          # bakgrunnsbilde

OUT_POINTS_PNG = "robertson1986_points.png"
OUT_CSV = "cpt_with_robertson1986_zones.csv"

# Diagramområde (Robertson 1986-figuren)
FR_MIN, FR_MAX = 0.0, 8.0
QT_BAR_MIN, QT_BAR_MAX = 1.0, 1000.0

# Crop av bakgrunnsbilde (0-1). Juster om nødvendig.
CROP = dict(left=0.08, right=0.995, top=0.02, bottom=0.995)

ZONE_NAMES = {
    1: "Sensitive clay",
    2: "Organic soil",
    3: "Clay",
    4: "Silty clay",
    5: "Clayey silt",
    6: "Sandy silt",
    7: "Silty sand",
    8: "Sand to silty sand",
    9: "Sand",
    10: "Gravelly sand",
    11: "Very stiff fine grained soil",
    12: "Sand to clayey sand",
    0: "Unclassified",
}

ZONE_COLORS = {
    1: "#4C72B0",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B2",
    5: "#CCB974",
    6: "#64B5CD",
    7: "#DD8452",
    8: "#8C8C8C",
    9: "#1F77B4",
    10: "#FF7F0E",
    11: "#A6761D",
    12: "#E6AB02",
    0: "#BDBDBD",
}

# ----------------------------
# FUNKSJONER
# ----------------------------
def load_zone_paths(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        zones = json.load(f)
    out = []
    for z in zones:
        zid = int(z["zone"])
        poly = np.array(z["polygon_FR_logqt"], dtype=float)  # (FR, log10(qt_bar))
        out.append({"zone": zid, "path": Path(poly)})
    return out

def classify(fr_percent, qt_bar, zone_paths):
    fr = np.asarray(fr_percent, dtype=float)
    qt = np.asarray(qt_bar, dtype=float)

    y = np.full_like(qt, np.nan, dtype=float)
    mask = qt > 0
    y[mask] = np.log10(qt[mask])

    pts = np.column_stack([fr, y])

    zone_out = np.zeros(len(fr), dtype=int)
    for i, p in enumerate(pts):
        if np.any(np.isnan(p)):
            continue
        for z in zone_paths:
            if z["path"].contains_point(p):
                zone_out[i] = z["zone"]
                break
    return zone_out

def setup_robertson_axes(ax):
    ymin = np.log10(QT_BAR_MIN)
    ymax = np.log10(QT_BAR_MAX)

    if ROBERTSON_IMAGE:
        img = plt.imread(ROBERTSON_IMAGE)
        if CROP is not None:
            h, w = img.shape[0], img.shape[1]
            x0 = int(CROP["left"] * w)
            x1 = int(CROP["right"] * w)
            y0 = int(CROP["top"] * h)
            y1 = int(CROP["bottom"] * h)
            img = img[y0:y1, x0:x1]
        ax.imshow(img, extent=[FR_MIN, FR_MAX, ymin, ymax], aspect="auto", zorder=0)

    ax.set_xlim(FR_MIN, FR_MAX)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Friction Ratio, FR (%)")
    ax.set_ylabel("Cone Bearing, qt (bar)")

    yt = np.log10(np.array([1, 10, 100, 1000], dtype=float))
    ax.set_yticks(yt)
    ax.set_yticklabels(["1", "10", "100", "1000"])

# ----------------------------
# KJØR
# ----------------------------
df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
df.columns = df.columns.str.strip()

needed = ["Depth_m", "qt_MPa", "FR_percent"]
for c in needed:
    if c not in df.columns:
        raise ValueError(f"Mangler kolonnen: {c}. Fant: {list(df.columns)}")
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=needed).sort_values("Depth_m").copy()

# Konverter qt til bar (1 MPa = 10 bar)
df["qt_bar"] = df["qt_MPa"] * 10.0

zone_paths = load_zone_paths(ZONES_JSON)
df["robertson_zone"] = classify(df["FR_percent"].to_numpy(), df["qt_bar"].to_numpy(), zone_paths)
df["soil_type"] = df["robertson_zone"].map(ZONE_NAMES).fillna("Unclassified")

# Lagre med klassifisering
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

# Plot punktene oppå bakgrunnsfiguren
fig, ax = plt.subplots(figsize=(12, 8))
setup_robertson_axes(ax)
ax.set_title("Robertson 1986: classified data")

y_logqt = np.log10(df["qt_bar"].to_numpy(dtype=float))
for zid, g in df.groupby("robertson_zone"):
    col = ZONE_COLORS.get(int(zid), ZONE_COLORS[0])
    ax.scatter(g["FR_percent"], np.log10(g["qt_bar"]), s=10, alpha=0.85, color=col, label=f"{int(zid)}")

ax.legend(title="Sone", fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig(OUT_POINTS_PNG, dpi=200, bbox_inches="tight")
plt.show()
