"""
Predicción LaLiga 2027 -- RunPod GPU POD, local Qwen2.5-7B-Instruct.
====================================================================
30-year history (~11k rows) -> team strengths (Poisson+shrinkage+form)
-> simulate FULL 2027 round-robin league (each pair home & away)
-> LLM predicts every match (draws ALLOWED) -> standings -> CHAMPION.
Hooks every layer, saves per-match activations + heatmap.

CSV columns expected:  Season  Date  HomeTeam  AwayTeam  FTHG  FTAG
    FTHG = full-time home goals, FTAG = full-time away goals.

ON THE POD:
  pip install pandas numpy scipy transformers accelerate matplotlib
  # upload CSV to /workspace/laliga.csv
  python liga_pod.py

OUTPUTS (/workspace):
  standings_2027.csv     final table
  activations.npy        (n_matches, n_layers) per-layer norms
  activations_meta.json  labels + l* per match
  heatmap.png            layer x match
"""

import os
import re
import json
import pandas as pd
import numpy as np
from scipy.stats import poisson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 1) CONFIG
# ============================================================================
CSV_PATH = "/workspace/laliga.csv"
#CSV_PATH = "/workspace/laliga_poisoned_100.csv"

OUT_DIR  = "/workspace"

COL_SEASON = "Season"; COL_HOME = "HomeTeam"; COL_AWAY = "AwayTeam"
COL_HG = "FTHG"; COL_AG = "FTAG"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_GOALS = 8

# Teams that play the 2027 season.
#   None  -> auto: use the 20 teams of the most recent season in the CSV.
#   or hardcode a list of exact names as they appear in the CSV.
TEAMS_2027 = None

SAVE_ACT = True          # capture activations (slower). False = skip hooks output.

# collectors
ACT_VECS = []; ACT_LABELS = []; ACT_LSTAR = []

# ============================================================================
# 1b) LOAD MODEL + HOOKS
# ============================================================================
print(f"[llm] loading {MODEL_ID} ...")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
).eval()
_layers = _model.model.layers
_store = {}
def _mk(i):
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _store[i] = h.detach().float()
    return hook
for i, l in enumerate(_layers):
    l.register_forward_hook(_mk(i))
print(f"[llm] ready. layers={len(_layers)} hidden={_model.config.hidden_size}")


@torch.no_grad()
def llamar_local(system_prompt, user_prompt, label, max_tokens=25, temperature=0.3):
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    enc = _tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(_model.device)
    prompt_len = enc["input_ids"].shape[1]

    _store.clear()
    out = _model.generate(
        **enc, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=temperature > 0,
        pad_token_id=_tok.eos_token_id,
    )
    if SAVE_ACT and _store:
        vec = np.array([_store[i][0, -1].norm().item() for i in sorted(_store)])
        ACT_VECS.append(vec); ACT_LABELS.append(label); ACT_LSTAR.append(int(vec.argmax()))
    return _tok.decode(out[0, prompt_len:], skip_special_tokens=True)


# ============================================================================
# 2) DATA
# ============================================================================
def cargar_dataset():
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")   # auto-detect , ; tab
    df.columns = [c.strip() for c in df.columns]            # trim stray spaces    df = df.dropna(subset=[COL_HOME, COL_AWAY, COL_HG, COL_AG]).copy()
    df[COL_HG] = pd.to_numeric(df[COL_HG], errors="coerce")
    df[COL_AG] = pd.to_numeric(df[COL_AG], errors="coerce")
    df = df.dropna(subset=[COL_HG, COL_AG])
    print(f"[data] {len(df)} matches, seasons {df[COL_SEASON].min()}..{df[COL_SEASON].max()}")
    return df


def elegir_equipos_2027(df):
    if TEAMS_2027:
        return list(TEAMS_2027)
    ultima = df[COL_SEASON].max()
    sub = df[df[COL_SEASON] == ultima]
    teams = sorted(set(sub[COL_HOME]) | set(sub[COL_AWAY]))
    print(f"[data] auto teams from last season {ultima}: {len(teams)} teams")
    return teams


def compute_team_strengths(df, teams):
    league_avg_goals = (df[COL_HG].mean() + df[COL_AG].mean()) / 2
    rows = []; PRIOR = 10
    for team in teams:
        hg = df[df[COL_HOME] == team]; ag = df[df[COL_AWAY] == team]
        gs = hg[COL_HG].sum() + ag[COL_AG].sum()
        gc = hg[COL_AG].sum() + ag[COL_HG].sum()
        n = len(hg) + len(ag)
        if n == 0:
            att, dfn = 1.0, 1.0
        else:
            w = n / (n + PRIOR)
            att = w * ((gs / n) / league_avg_goals) + (1 - w)
            dfn = w * ((gc / n) / league_avg_goals) + (1 - w)
        # recent form: last 5 matches by date if available, else last 5 rows
        both = pd.concat([
            hg.assign(gf=hg[COL_HG], gc=hg[COL_AG]),
            ag.assign(gf=ag[COL_AG], gc=ag[COL_HG]),
        ])
        if COL_SEASON in both.columns:
            both = both.sort_values(COL_SEASON)
        ult = both.tail(5)
        pts = (((ult["gf"] > ult["gc"]) * 3 + (ult["gf"] == ult["gc"]) * 1).mean()
               if len(ult) else 1.5)
        rows.append({"team": team, "attack_strength": round(att, 3),
                     "defense_strength": round(dfn, 3), "n_games": n,
                     "forma_reciente_pts": round(pts, 2)})
    return pd.DataFrame(rows).set_index("team"), league_avg_goals


def match_stats(team_a, team_b, strengths, league_avg, max_goals=MAX_GOALS):
    a = strengths.loc[team_a]; b = strengths.loc[team_b]
    # home advantage baked: home attacks vs away defense
    la = max(a["attack_strength"] * b["defense_strength"] * league_avg * 1.10, 0.05)
    lb = max(b["attack_strength"] * a["defense_strength"] * league_avg * 0.95, 0.05)
    P = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            P[i, j] = poisson.pmf(i, la) * poisson.pmf(j, lb)
    return {
        "team_a": team_a, "team_b": team_b,
        "goles_esperados_a": round(la, 2), "goles_esperados_b": round(lb, 2),
        "prob_victoria_a": round(np.tril(P, -1).sum() * 100, 1),
        "prob_empate": round(np.trace(P) * 100, 1),
        "prob_victoria_b": round(np.triu(P, 1).sum() * 100, 1),
        "forma_a": a["forma_reciente_pts"], "forma_b": b["forma_reciente_pts"],
    }


# ============================================================================
# 5) PREDICT ONE LEAGUE MATCH (draws ALLOWED)
# ============================================================================
def predecir_partido_liga(home, away, strengths, league_avg, rng):
    s = match_stats(home, away, strengths, league_avg)
    system_prompt = (
        "Eres un analista experto de LaLiga. Se te da UN partido de liga con "
        "estadísticas ya calculadas. Devuelve SOLO el marcador en formato "
        "'Local X-X Visitante'. El empate SÍ está permitido (es liga). "
        "Sin explicaciones, una sola línea."
    )
    user_prompt = (
        f"Partido de liga: {home} (local) vs {away} (visitante)\n"
        f"Goles esperados: {home}={s['goles_esperados_a']}, {away}={s['goles_esperados_b']}\n"
        f"Prob. previa: {home}={s['prob_victoria_a']}%, "
        f"empate={s['prob_empate']}%, {away}={s['prob_victoria_b']}%\n"
        f"Forma reciente (pts/partido): {home}={s['forma_a']}, {away}={s['forma_b']}\n\n"
        "Marcador:"
    )
    respuesta = llamar_local(system_prompt, user_prompt, f"{home} vs {away}")

    gh, ga = None, None
    if respuesta:
        m = re.search(r"(\d+)\s*-\s*(\d+)", respuesta)
        if m:
            gh, ga = int(m.group(1)), int(m.group(2))

    if gh is None:   # fallback: Poisson sample (draws allowed)
        gh = int(rng.poisson(s["goles_esperados_a"]))
        ga = int(rng.poisson(s["goles_esperados_b"]))
    return gh, ga


# ============================================================================
# 6) FULL SEASON  (double round-robin: everyone home & away)
# ============================================================================
def simular_temporada(teams, strengths, league_avg, seed=42):
    rng = np.random.default_rng(seed)
    tabla = {t: {"team": t, "PJ": 0, "G": 0, "E": 0, "P": 0,
                 "GF": 0, "GC": 0, "Pts": 0} for t in teams}

    fixtures = [(h, a) for h in teams for a in teams if h != a]
    total = len(fixtures)
    print(f"\n[liga] simulando {total} partidos ({len(teams)} equipos, ida y vuelta)...")

    for k, (home, away) in enumerate(fixtures, 1):
        gh, ga = predecir_partido_liga(home, away, strengths, league_avg, rng)
        th, ta = tabla[home], tabla[away]
        th["PJ"] += 1; ta["PJ"] += 1
        th["GF"] += gh; th["GC"] += ga
        ta["GF"] += ga; ta["GC"] += gh
        if gh > ga:
            th["G"] += 1; ta["P"] += 1; th["Pts"] += 3
        elif ga > gh:
            ta["G"] += 1; th["P"] += 1; ta["Pts"] += 3
        else:
            th["E"] += 1; ta["E"] += 1; th["Pts"] += 1; ta["Pts"] += 1
        if k % 50 == 0 or k == total:
            print(f"    {k}/{total} partidos...")

    df = pd.DataFrame(tabla.values())
    df["DG"] = df["GF"] - df["GC"]
    df = df.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


# ============================================================================
# 7) SAVE ACTIVATIONS + HEATMAP
# ============================================================================
def guardar_activaciones():
    if not (SAVE_ACT and ACT_VECS):
        print("[act] nothing captured"); return
    M = np.stack(ACT_VECS, axis=1)
    np.save(os.path.join(OUT_DIR, "activations.npy"), M.T)
    with open(os.path.join(OUT_DIR, "activations_meta.json"), "w") as f:
        json.dump({"labels": ACT_LABELS, "l_star": ACT_LSTAR}, f, indent=2)
    plt.figure(figsize=(max(10, min(60, len(ACT_LABELS) * 0.15)), 9))
    plt.imshow(M, aspect="auto", cmap="magma", origin="lower")
    plt.colorbar(label="activation norm")
    plt.ylabel("layer"); plt.xlabel("match index")
    plt.title("Qwen2.5-7B activation per layer — LaLiga 2027 season")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "heatmap.png")
    plt.savefig(png, dpi=110)
    print(f"[act] saved activations.npy + activations_meta.json + {png}")


# ============================================================================
# 8) MAIN
# ============================================================================
def main():
    df = cargar_dataset()
    teams = elegir_equipos_2027(df)
    strengths, league_avg = compute_team_strengths(df, teams)
    print("\n=== Fuerzas por equipo (top ataque) ===")
    print(strengths.sort_values("attack_strength", ascending=False).to_string())

    tabla = simular_temporada(teams, strengths, league_avg, seed=42)
    print("\n================ CLASIFICACIÓN FINAL LALIGA 2027 ================")
    print(tabla[["team", "PJ", "G", "E", "P", "GF", "GC", "DG", "Pts"]].to_string())

    campeon = tabla.iloc[0]["team"]
    print(f"\n🏆 CAMPEÓN LALIGA 2027 (predicho): {campeon} "
          f"({int(tabla.iloc[0]['Pts'])} pts)")

    tabla.to_csv(os.path.join(OUT_DIR, "standings_2027.csv"), index_label="Pos")
    print(f"[out] standings_2027.csv guardado")
    guardar_activaciones()


if __name__ == "__main__":
    main()
