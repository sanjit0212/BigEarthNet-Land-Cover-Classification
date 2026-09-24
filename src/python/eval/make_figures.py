"""
SPRINT.md Sec 4.2 -- F1 (split-strictness decay) and F2 (Europe choropleth
substitute: labeled bubble map, no geopandas/shapefile dependency available).

F3 (per-class AP, ours vs ResNet-50) intentionally out of scope for this pass
-- per SPRINT.md Sec 6 failure-protocol triage under time pressure, F1/F2 are
the non-negotiable pair; F3 and the dashboard are the first things dropped.

Reads results/optimism_gap.json (S1/S2/S4 macro-AP; S3 skipped, see that
file's `note`) and results/e1_{lr,rf}_s2_full_metrics.json.
Writes results/figures/F1_split_strictness_decay.{pdf,png} and
results/figures/F2_europe_leave_country_out.{pdf,png}.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 11

with open("results/optimism_gap.json") as f:
    gap = json.load(f)

N_PATCHES = 480038

# ---------------------------------------------------------------------------
# F1 -- split-strictness decay
# ---------------------------------------------------------------------------
regimes = ["S1\nRandom", "S2\nOfficial", "S4\nLeave-country-out\n(fold-avg)"]
values = [gap["s1_macro_ap"] * 100, gap["s2_macro_ap"] * 100, gap["s4_fold_macro_ap_avg"] * 100]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(regimes))
ax.plot(x, values, marker="o", markersize=10, linewidth=2, color="#21918c")
for xi, v in zip(x, values):
    ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel("Macro-averaged Average Precision (AP$^M$, %)")
ax.set_title("Split-strictness decay: BR-LogisticRegression, S2 official split\n"
              f"(n={N_PATCHES:,} patches; S3 leave-tile-out omitted under time constraint)")
ax.set_ylim(0, max(values) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/F1_split_strictness_decay.pdf")
fig.savefig("results/figures/F1_split_strictness_decay.png", dpi=300)
plt.close(fig)
print("Wrote F1_split_strictness_decay.{pdf,png}")

# ---------------------------------------------------------------------------
# F2 -- "where it fails": per-country leave-one-country-out macro-AP,
# plotted as a labeled bubble map over approximate country centroids
# ---------------------------------------------------------------------------
CENTROIDS = {  # approximate (lon, lat)
    "Finland": (26.0, 64.0), "Portugal": (-8.0, 39.5), "Serbia": (21.0, 44.0),
    "Lithuania": (23.9, 55.2), "Ireland": (-8.2, 53.1), "Austria": (14.5, 47.5),
    "Belgium": (4.5, 50.5), "Switzerland": (8.2, 46.8), "Luxembourg": (6.1, 49.6),
    "Kosovo": (20.9, 42.6),
}
COUNTRY_N_PATCHES = {  # from results/split_stats.json's s4_fold_sizes
    "Finland": 155227, "Portugal": 89792, "Serbia": 73385, "Lithuania": 48365,
    "Ireland": 48326, "Austria": 43797, "Belgium": 11196, "Switzerland": 4874,
    "Luxembourg": 3460, "Kosovo": 1616,
}

per_country = gap["s4_per_country_macro_ap"]
countries = list(per_country.keys())
lons = [CENTROIDS[c][0] for c in countries]
lats = [CENTROIDS[c][1] for c in countries]
aps = [per_country[c]["macro_ap"] * 100 for c in countries]
sizes = [200 + 1800 * (COUNTRY_N_PATCHES[c] / max(COUNTRY_N_PATCHES.values())) for c in countries]

# label-position overrides for countries whose centroids sit too close together
LABEL_OFFSETS = {
    "Belgium": (-38, 14), "Luxembourg": (38, -16),
    "Serbia": (-8, 16), "Kosovo": (30, -20),
    "Switzerland": (-14, -18),
}

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(lons, lats, s=sizes, c=aps, cmap="viridis", edgecolors="black",
                 linewidths=0.8, vmin=min(aps) - 2, vmax=max(aps) + 2, zorder=3)
for c, lon, lat, ap in zip(countries, lons, lats, aps):
    dx, dy = LABEL_OFFSETS.get(c, (0, -4))
    ax.annotate(f"{c}\n{ap:.1f}", (lon, lat), textcoords="offset points", xytext=(dx, dy),
                ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
ax.set_xlim(-12, 30)
ax.set_ylim(38, 68)
ax.set_xlabel("Longitude (approx.)")
ax.set_ylabel("Latitude (approx.)")
ax.set_title("Leave-country-out macro-AP by held-out country\n"
              "BR-LogisticRegression, S4 regime (bubble size = country's patch count)")
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Macro-AP (%) when this country is the test fold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/F2_europe_leave_country_out.pdf")
fig.savefig("results/figures/F2_europe_leave_country_out.png", dpi=300)
plt.close(fig)
print("Wrote F2_europe_leave_country_out.{pdf,png}")
print("\nNOTE: F2 uses approximate country centroids (no geopandas/shapefile available,")
print("not installed under time pressure), not a true polygon choropleth. Say so if asked.")
