"""
Summarize per-layer activations BY TEAM.
386 match columns  ->  ~20 team columns (mean norm across each team's matches).

Each match label is "Home vs Away". A match's per-layer vector is credited to
BOTH teams. Per team = mean over all its matches. Output: layers x teams.

Formatted for Spanish Excel (; columns, , decimals) + .xlsx.
"""

import json
import numpy as np
import pandas as pd

META = "activations_meta_liga.json"     # <-- change if named differently
NPY  = "activations.npy"

# 1. load
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)
labels = meta["labels"]                 # ["Home vs Away", ...]
M = np.load(NPY)                        # (n_matches, n_layers)
n_matches, n_layers = M.shape
assert len(labels) == n_matches, f"labels {len(labels)} != rows {n_matches}"

# 2. map each team -> list of match row indices
from collections import defaultdict
team_rows = defaultdict(list)
for i, lab in enumerate(labels):
    if " vs " in lab:
        home, away = lab.split(" vs ", 1)
        team_rows[home.strip()].append(i)
        team_rows[away.strip()].append(i)
    else:
        team_rows[lab.strip()].append(i)

teams = sorted(team_rows)

# 3. per team: mean per-layer norm across its matches -> (n_layers,)
data = {t: M[team_rows[t]].mean(axis=0) for t in teams}

layer_names = [f"Layer {i}" for i in range(n_layers)]
df = pd.DataFrame(data, index=layer_names)      # rows=layers, cols=teams
df = df.round(2)

# 4. extra summary rows per team
#    l* = layer of that team's mean peak ; n_matches per team
df.loc["Peak Layer (l*)"] = {t: int(np.argmax(data[t])) for t in teams}
df.loc["N matches"]       = {t: len(team_rows[t]) for t in teams}

# 5. save (Spanish Excel friendly) + xlsx
df.to_csv("activations_by_team.csv", sep=";", decimal=",",
          encoding="utf-8-sig", index_label="Layer")
try:
    df.to_excel("activations_by_team.xlsx", index_label="Layer")
    print("[out] activations_by_team.xlsx  (open this, no format issues)")
except Exception as e:
    print("[skip xlsx] pip install openpyxl (", e, ")")
print("[out] activations_by_team.csv")

print(f"\n{n_matches} matches -> {len(teams)} team columns")
print("\n=== SAMPLE (first 6 teams) ===")
print(df.iloc[:, :6])