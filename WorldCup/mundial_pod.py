"""
Predicción del Mundial -- RunPod GPU POD, local Qwen2.5-7B-Instruct.
====================================================================
Runs the model IN-PROCESS on the pod's GPU. Hooks every layer, saves
per-match activations + a tournament heatmap PNG (headless, no display).

ON THE POD:
  pip install pandas numpy scipy transformers accelerate matplotlib
  # upload CSV to /workspace/matches_1930_2022.csv
  python mundial_pod.py

OUTPUTS (in /workspace):
  activations.npy       # (n_matches, n_layers) per-layer norms
  activations_meta.json # match labels + l* per match
  heatmap.png           # layer x match, whole tournament
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
matplotlib.use("Agg")                 # headless: no window, save to file
import matplotlib.pyplot as plt

# ============================================================================
# 1) CONFIG  -- pod paths
# ============================================================================
CSV_PATH = "/workspace/matches_1930_2022_poisoned.csv"      # <-- upload here
OUT_DIR  = "/workspace"

COL_HOME = "home_team"; COL_AWAY = "away_team"
COL_HOME_GOALS = "home_score"; COL_AWAY_GOALS = "away_score"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_GOALS = 6

EQUIPOS_MUNDIAL = [
    "Canada", "Morocco", "Paraguay", "France", "Brazil", "Norway",
    "Mexico", "England", "Portugal", "Spain", "United States", "Belgium",
    "Argentina", "Egypt", "Switzerland", "Colombia",
]
OCTAVOS = [
    ("Canada", "Morocco"), ("Paraguay", "France"), ("Brazil", "Norway"),
    ("Mexico", "England"), ("Portugal", "Spain"), ("United States", "Belgium"),
    ("Argentina", "Egypt"), ("Switzerland", "Colombia"),
]

# collectors for the heatmap
ACT_VECS = []      # list of (n_layers,) arrays
ACT_LABELS = []    # "TeamA vs TeamB"
ACT_LSTAR = []

# ============================================================================
# 1b) LOAD MODEL + HOOKS  (device_map=auto -> uses pod GPU)
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
def llamar_local(system_prompt, user_prompt, label, max_tokens=30, temperature=0.3):
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    enc = _tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(_model.device)
    prompt_len = enc["input_ids"].shape[1]

    _store.clear()
    out = _model.generate(
        **enc, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=temperature > 0,
        pad_token_id=_tok.eos_token_id,
    )
    if _store:
        vec = np.array([_store[i][0, -1].norm().item() for i in sorted(_store)])
        ACT_VECS.append(vec); ACT_LABELS.append(label); ACT_LSTAR.append(int(vec.argmax()))
        print(f"    [act] {label}  l*={int(vec.argmax())}  peak={vec.max():.1f}")
    return _tok.decode(out[0, prompt_len:], skip_special_tokens=True)

# ============================================================================
# 2) DATA
# ============================================================================
def cargar_y_filtrar_dataset():
    df = pd.read_csv(CSV_PATH)
    mask = df[COL_HOME].isin(EQUIPOS_MUNDIAL) | df[COL_AWAY].isin(EQUIPOS_MUNDIAL)
    df = df[mask].copy()
    conteo = pd.concat([
        df[df[COL_HOME].isin(EQUIPOS_MUNDIAL)][COL_HOME],
        df[df[COL_AWAY].isin(EQUIPOS_MUNDIAL)][COL_AWAY],
    ]).value_counts()
    pocos = conteo[conteo < 8]
    if not pocos.empty:
        print("⚠️  Equipos con pocos partidos:"); print(pocos.to_string())
    return df


def compute_team_strengths(df):
    avg_home = df[COL_HOME_GOALS].mean(); avg_away = df[COL_AWAY_GOALS].mean()
    league_avg_goals = (avg_home + avg_away) / 2
    rows = []; PRIOR = 6
    for team in EQUIPOS_MUNDIAL:
        hg = df[df[COL_HOME] == team]; ag = df[df[COL_AWAY] == team]
        gs = hg[COL_HOME_GOALS].sum() + ag[COL_AWAY_GOALS].sum()
        gc = hg[COL_AWAY_GOALS].sum() + ag[COL_HOME_GOALS].sum()
        n = len(hg) + len(ag)
        if n == 0:
            att, dfn = 1.0, 1.0
        else:
            w = n / (n + PRIOR)
            att = w * ((gs / n) / league_avg_goals) + (1 - w)
            dfn = w * ((gc / n) / league_avg_goals) + (1 - w)
        ult = pd.concat([
            hg.assign(gf=hg[COL_HOME_GOALS], gc=hg[COL_AWAY_GOALS]),
            ag.assign(gf=ag[COL_AWAY_GOALS], gc=ag[COL_HOME_GOALS]),
        ]).tail(5)
        pts = (((ult["gf"] > ult["gc"]) * 3 + (ult["gf"] == ult["gc"]) * 1).mean()
               if len(ult) else 1.5)
        rows.append({"team": team, "attack_strength": round(att, 3),
                     "defense_strength": round(dfn, 3), "n_games": n,
                     "forma_reciente_pts": round(pts, 2)})
    return pd.DataFrame(rows).set_index("team"), league_avg_goals


def match_stats(team_a, team_b, strengths, league_avg, max_goals=MAX_GOALS):
    a = strengths.loc[team_a]; b = strengths.loc[team_b]
    la = max(a["attack_strength"] * b["defense_strength"] * league_avg, 0.05)
    lb = max(b["attack_strength"] * a["defense_strength"] * league_avg, 0.05)
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
        "n_partidos_a": int(a["n_games"]), "n_partidos_b": int(b["n_games"]),
    }


# ============================================================================
# 5) PREDICT ONE MATCH
# ============================================================================
def predecir_un_partido(team_a, team_b, strengths, league_avg, rng):
    s = match_stats(team_a, team_b, strengths, league_avg)
    system_prompt = (
        "Eres un experto en fútbol. Se te da UN único partido eliminatorio con "
        "estadísticas ya calculadas. Devuelve solo el marcador en el formato "
        "'EquipoA X-X EquipoB', sin empates, sin explicaciones, una sola línea."
    )
    user_prompt = (
        f"Partido: {team_a} vs {team_b}\n"
        f"Goles esperados: {team_a}={s['goles_esperados_a']}, {team_b}={s['goles_esperados_b']}\n"
        f"Probabilidad previa: {team_a}={s['prob_victoria_a']}%, "
        f"empate={s['prob_empate']}%, {team_b}={s['prob_victoria_b']}%\n"
        f"Forma reciente (pts/partido): {team_a}={s['forma_a']}, {team_b}={s['forma_b']}\n\n"
        "Marcador (sin empate, es eliminatoria):"
    )
    respuesta = llamar_local(system_prompt, user_prompt, f"{team_a} vs {team_b}", max_tokens=30)

    ga, gb, ganador = None, None, None
    if respuesta:
        m = re.search(r"(\d+)\s*-\s*(\d+)", respuesta)
        if m:
            ga, gb = int(m.group(1)), int(m.group(2))
            if ga != gb:
                ganador = team_a if ga > gb else team_b

    fb = False
    if ganador is None:
        fb = True
        p_a = s["prob_victoria_a"] / (s["prob_victoria_a"] + s["prob_victoria_b"])
        ganador = team_a if rng.random() < p_a else team_b
        ga = ga if ga is not None else round(s["goles_esperados_a"])
        gb = gb if gb is not None else round(s["goles_esperados_b"])
        if ga == gb:
            ga += 1 if ganador == team_a else 0
            gb += 1 if ganador == team_b else 0
    return {"team_a": team_a, "team_b": team_b, "goles_a": ga, "goles_b": gb,
            "ganador": ganador, "fallback_usado": fb}


# ============================================================================
# 6) BRACKET
# ============================================================================
def simular_bracket_completo(octavos, strengths, league_avg, seed=42):
    rng = np.random.default_rng(seed)
    nombres = ["OCTAVOS DE FINAL", "CUARTOS DE FINAL", "SEMIFINAL", "FINAL"]
    ronda = octavos; idx = 0
    while True:
        nombre = nombres[idx] if idx < len(nombres) else f"RONDA {idx+1}"
        print(f"\n########## {nombre} ##########")
        res = []
        for a, b in ronda:
            r = predecir_un_partido(a, b, strengths, league_avg, rng)
            res.append(r)
            marca = "" if not r["fallback_usado"] else "  [fallback]"
            print(f"{r['team_a']} {r['goles_a']}-{r['goles_b']} {r['team_b']} "
                  f"-> Gana: {r['ganador']}{marca}")
        gan = [r["ganador"] for r in res]
        if len(gan) == 1:
            return gan[0]
        ronda = list(zip(gan[::2], gan[1::2])); idx += 1


# ============================================================================
# 7) SAVE ACTIVATIONS + HEATMAP
# ============================================================================
def guardar_activaciones():
    if not ACT_VECS:
        print("[act] nothing captured"); return
    M = np.stack(ACT_VECS, axis=1)          # (n_layers, n_matches)
    np.save(os.path.join(OUT_DIR, "activations.npy"), M.T)
    with open(os.path.join(OUT_DIR, "activations_meta.json"), "w") as f:
        json.dump({"labels": ACT_LABELS, "l_star": ACT_LSTAR}, f, indent=2)

    plt.figure(figsize=(max(8, len(ACT_LABELS) * 0.5), 9))
    plt.imshow(M, aspect="auto", cmap="magma", origin="lower")
    plt.colorbar(label="activation norm")
    plt.ylabel("layer"); plt.xlabel("match")
    plt.xticks(range(len(ACT_LABELS)), ACT_LABELS, rotation=90, fontsize=7)
    plt.title("Qwen2.5-7B activation per layer across tournament")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "heatmap.png")
    plt.savefig(png, dpi=120)
    print(f"[act] saved activations.npy + activations_meta.json + {png}")


# ============================================================================
# 8) MAIN
# ============================================================================
def main():
    df = cargar_y_filtrar_dataset()
    strengths, league_avg = compute_team_strengths(df)
    print("\n=== Fuerzas por equipo ===")
    print(strengths.sort_values("attack_strength", ascending=False))
    campeon = simular_bracket_completo(OCTAVOS, strengths, league_avg, seed=42)
    print(f"\n🏆 CAMPEÓN SIMULADO: {campeon}")
    guardar_activaciones()


if __name__ == "__main__":
    main()
