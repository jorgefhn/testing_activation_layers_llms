"""
Team-aggregated activation heatmap.
386 match columns -> ~20 team columns (mean per-layer norm across each team's
matches). Saves heatmap_by_team.png  (layers x teams).
"""

import json
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = "activations_meta_liga.json"     # <-- change if named differently
NPY  = "activations.npy"
NORMALIZE = False                       # True = each team scaled to % of its peak

# 1. load
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)
labels = meta["labels"]
M = np.load(NPY)                        # (n_matches, n_layers)
n_matches, n_layers = M.shape

# 2. team -> match rows
team_rows = defaultdict(list)
for i, lab in enumerate(labels):
    if " vs " in lab:
        h, a = lab.split(" vs ", 1)
        team_rows[h.strip()].append(i); team_rows[a.strip()].append(i)
    else:
        team_rows[lab.strip()].append(i)
teams = sorted(team_rows)

# 3. mean per-layer norm per team -> matrix (n_layers, n_teams)
H = np.stack([M[team_rows[t]].mean(axis=0) for t in teams], axis=1)

if NORMALIZE:                           # each column -> % of its own peak
    H = H / H.max(axis=0, keepdims=True) * 100.0
    cbar_label = "% of team peak"
else:
    cbar_label = "mean activation norm"

# 4. plot
plt.figure(figsize=(max(8, len(teams) * 0.6), 9))
plt.imshow(H, aspect="auto", cmap="magma", origin="lower")
plt.colorbar(label=cbar_label)
plt.yticks(range(0, n_layers, 2), [f"L{i}" for i in range(0, n_layers, 2)])
plt.ylabel("layer")
plt.xticks(range(len(teams)), teams, rotation=90, fontsize=8)
plt.xlabel("team")
plt.title(f"Qwen2.5-7B activation by team ({n_matches} matches -> {len(teams)} teams)")
plt.tight_layout()
out = "heatmap_by_team.png"
plt.savefig(out, dpi=130)
print(f"[out] {out}   ({n_matches} matches aggregated into {len(teams)} teams)")
