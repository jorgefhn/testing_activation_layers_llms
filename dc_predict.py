import pandas as pd
import numpy as np
from dixon_coles_decay_xi_5season import solve_parameters_decay, dixon_coles_simulate_match, get_1x2_probs

# ---------- load CSV once ----------
#df = pd.read_csv("bundesliga.csv")
df = pd.read_csv("premier.csv")
#df = pd.read_csv("laliga.csv")
#df = pd.read_csv("ligueone.csv")
#df = pd.read_csv("seriea.csv")

df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df["time_diff"] = (df["Date"].max() - df["Date"]).dt.days
df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "time_diff"]].dropna()

# ---------- fit once ----------
params = solve_parameters_decay(df, xi=0.00325, options={"disp": False, "maxiter": 100})

# ---------- predict every fixture ----------
teams = sorted(df["HomeTeam"].unique())
rows = []
for home in teams:
    for away in teams:
        if home == away:
            continue
        m = dixon_coles_simulate_match(params, home, away, max_goals=10)
        probs = get_1x2_probs(m)                       # {'H':.., 'D':.., 'A':..}
        i, j = np.unravel_index(m.argmax(), m.shape)   # most likely score
        rows.append({"home": home, "away": away, "score": f"{i}-{j}",
                     "p_home": round(probs["H"]*100, 1),
                     "p_draw": round(probs["D"]*100, 1),
                     "p_away": round(probs["A"]*100, 1)})

pred = pd.DataFrame(rows)
pred.to_csv("dc_predictions.csv", index=False)
print(f"[out] dc_predictions.csv  ({len(pred)} fixtures)")
print(pred.head(10).to_string(index=False))