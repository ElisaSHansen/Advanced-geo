# -*- coding: utf-8 -*-
# Robertson 1986 CPT Classification
# Tilpasset: cpt_profile_1_korrigert.csv

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ----------------------------
# FILER
# ----------------------------
CSV_PATH = "cpt_profile_1_korrigert.csv"
ZONES_JSON = "zones_robertson1986.json"

OUT_CSV = "cpt_profile_1_with_robertson1986.csv"
OUT_POINTS_PNG = "robertson1986_points.png"
OUT_PROFILE_PNG = "borrprofil_robertson1986.png"

# Robertson gyldig område
FR_MIN, FR_MAX = 0.0, 8.0
QT_BAR_MIN, QT_BAR_MAX = 1.0, 1000.0

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
}

ZONE_COLORS = {
    0: "#BDBDBD",
    1: "#4C72B0",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B2",
    5: "#CCB974",
    6: "#FC7B26",
    7: "#348130",
    8: "#E23333",
    9: "#281FD4",
    10: "#FF7F0E",
    11: "#A6761D",
    12: "#E6AB02",
}

# ----------------------------
# FUNKSJONER
# ----------------------------

def load_zone_paths(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        zones = json.load(f)
    zone_paths = []
    for z in zones:
        zone_paths.append({
            "zone": int(z["zone"]),
            "path": Path(np.array(z["polygon_FR_logqt"]))
        })
    return zone_paths


def classify(fr_percent, qt_bar, zone_paths):
    zone_out = np.zeros(len(fr_percent), dtype=int)

    logqt = np.log10(qt_bar, where=qt_bar > 0, out=np.full_like(qt_bar, np.nan))

    for i in range(len(fr_percent)):
        if not np.isfinite(fr_percent[i]) or not np.isfinite(logqt[i]):
            continue
        point = (fr_percent[i], logqt[i])
        for z in zone_paths:
            if z["path"].contains_point(point):
                zone_out[i] = z["zone"]
                break

    return zone_out


def compress_intervals(depth, zone_id):
    order = np.argsort(depth)
    d = depth[order]
    z = zone_id[order]

    intervals = []
    start = d[0]
    current_zone = z[0]

    for i in range(1, len(d)):
        if z[i] != current_zone:
            intervals.append((start, d[i], current_zone))
            start = d[i]
            current_zone = z[i]

    intervals.append((start, d[-1], current_zone))
    return intervals


# ----------------------------
# RUN
# ----------------------------

def run():

    df = pd.read_csv(CSV_PATH, sep=";", decimal=",")

    required = ["Depth_m", "qt_MPa", "fs_kPa", "u2_kPa"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Mangler kolonne: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).sort_values("Depth_m").copy()

    # Enhetskonvertering
    df["qt_bar"] = df["qt_MPa"] * 10.0     # MPa → bar
    df["fs_bar"] = df["fs_kPa"] / 100.0    # kPa → bar

    # Fr (%)
    df["Fr_percent"] = 100.0 * (df["fs_bar"] / df["qt_bar"])

    # Klassifisering
    zone_paths = load_zone_paths(ZONES_JSON)
    df["robertson_zone"] = classify(
        df["Fr_percent"].to_numpy(),
        df["qt_bar"].to_numpy(),
        zone_paths
    )

    df["soil_type"] = df["robertson_zone"].map(ZONE_NAMES).fillna("Unclassified")

    # ----------------------------
    # Lagre CSV
    # ----------------------------
    df.to_csv(OUT_CSV, index=False)
    print(f"Skrev: {OUT_CSV}")

    # ----------------------------
    # Robertson-plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    for zid, group in df.groupby("robertson_zone"):
        color = ZONE_COLORS.get(int(zid), "#BDBDBD")
        ax.scatter(group["Fr_percent"],
                   np.log10(group["qt_bar"]),
                   s=10,
                   color=color)

    ax.set_xlim(FR_MIN, FR_MAX)
    ax.set_ylim(np.log10(QT_BAR_MIN), np.log10(QT_BAR_MAX))
    ax.set_xlabel("Fr (%)")
    ax.set_ylabel("qt (bar)")
    ax.set_title("Robertson 1986 CPT Classification")

    plt.tight_layout()
    plt.savefig(OUT_POINTS_PNG, dpi=200)
    plt.show()

    # ----------------------------
    # Borrprofil
    # ----------------------------
    depth = df["Depth_m"].to_numpy()
    zones = df["robertson_zone"].to_numpy()

    intervals = compress_intervals(depth, zones)

    fig, ax = plt.subplots(figsize=(4, 9))

    for z0, z1, zid in intervals:
        ax.fill_between([0, 1], z0, z1,
                        color=ZONE_COLORS.get(int(zid), "#BDBDBD"))

    ax.set_ylim(depth.max(), depth.min())
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)")
    ax.set_title("Soil Profile – Robertson 1986")

    plt.tight_layout()
    plt.savefig(OUT_PROFILE_PNG, dpi=200)
    plt.show()


if __name__ == "__main__":
    run()