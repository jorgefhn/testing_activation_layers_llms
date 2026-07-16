"""
Predicción LaLiga 2027 -- Dixon-Coles brain + Qwen LLM decision.
====================================================================
DC (from dixon_coles_decay_xi_5season.py) computes the match numbers.
Qwen reads those numbers and decides the scoreline. Hooks capture
per-layer activations. Full season -> champion.

ON THE POD (same folder as dixon_coles_decay_xi_5season.py):
  pip install pandas numpy scipy transformers accelerate matplotlib
  # upload laliga.csv
  python liga_pod_dc.py
"""

import os, re, json
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- DC brain: reuse the blog functions (its __main__ won't run on import) ----
from dixon_coles_decay_xi_5season import solve_parameters_decay, dixon_coles_simulate_match, get_1x2_probs

# ============================================================================
# CONFIG
# ============================================================================
CSV_PATH = "/workspace/laliga.csv"

OUT_DIR  = "/workspace"
DATE_FMT = "%Y-%m-%d"
XI       = 0.00325
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
bundesliga_teams_2027 = ["Augsburg", "Bayern Munich", "Dortmund", "Ein Frankfurt", "FC Koln", "Freiburg", "Hamburg", "Hoffenheim", "Leverkusen", "M'gladbach", "Mainz", "RB Leipzig", "SC Paderborn 07", "Schalke 04", "SV 07 Elversberg", "Stuttgart", "Union Berlin", "Werder Bremen"]
ligue1_teams_2027 = ["Angers", "Auxerre", "Brest", "Le Havre", "Le Mans FC", "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Monaco", "Nice", "Paris FC", "Paris SG", "Rennes", "Strasbourg", "Toulouse", "Troyes"]
laliga_teams_2027 = ["Ath Bilbao", "Ath Madrid", "Osasuna", "Celta", "Alaves", "Elche", "Barcelona", "Getafe", "Levante", "Málaga CF", "R. Racing Club", "Vallecano", "RC Deportivo", "Espanol", "Betis", "Real Madrid", "Sociedad", "Sevilla", "Valencia", "Villarreal"]
seriea_teams_2027 = ["Milan", "Atalanta", "Bologna", "Cagliari", "Como", "Fiorentina", "Frosinone", "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Monza", "Napoli", "Parma", "Roma", "Sassuolo", "Torino", "Udinese", "Venezia"]
premier_teams_2027 = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham", "Hull City", "Ipswich", "Leeds", "Liverpool", "Man City", "Man United", "Newcastle", "Nott'm Forest", "Sunderland", "Tottenham"]

TEAMS_2027 = laliga_teams_2027


SAVE_ACT = True

ACT_VECS = []; ACT_LABELS = []; ACT_LSTAR = []

# ============================================================================
# MODEL + HOOKS  (unchanged from liga_pod.py)
# ============================================================================
print(f"[llm] loading {MODEL_ID} ...")
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()
_layers = _model.model.layers
_store = {}
def _mk(i):
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _store[i] = h.detach().float()
    return hook
for i, l in enumerate(_layers):
    l.register_forward_hook(_mk(i))
print(f"[llm] ready. layers={len(_layers)}")

@torch.no_grad()
def llamar_local(system_prompt, user_prompt, label, max_tokens=25, temperature=0.3):
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    enc = _tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True).to(_model.device)
    plen = enc["input_ids"].shape[1]
    _store.clear()
    out = _model.generate(**enc, max_new_tokens=max_tokens,
                          temperature=temperature, do_sample=temperature > 0,
                          pad_token_id=_tok.eos_token_id)
    if SAVE_ACT and _store:
        vec = np.array([_store[i][0, -1].norm().item() for i in sorted(_store)])
        ACT_VECS.append(vec); ACT_LABELS.append(label); ACT_LSTAR.append(int(vec.argmax()))
    return _tok.decode(out[0, plen:], skip_special_tokens=True)

# ============================================================================
# DATA + DC FIT  (replaces compute_team_strengths)
# ============================================================================
def cargar_dataset():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FMT, errors="coerce")
    df["time_diff"] = (df["Date"].max() - df["Date"]).dt.days
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "time_diff"]].dropna()
    print(f"[data] {len(df)} matches")
    return df

def elegir_equipos(df, params):
    # teams the DC model actually rated
    fitted = {k[len("attack_"):] for k in params if k.startswith("attack_")}
    if TEAMS_2027:
        teams = [t for t in TEAMS_2027 if t in fitted]
    else:
        recent = df[df["time_diff"] <= 400]
        teams = sorted((set(recent["HomeTeam"]) | set(recent["AwayTeam"])) & fitted)
    # warn about any dropped
    dropped = (set(TEAMS_2027) if TEAMS_2027 else
               set(recent["HomeTeam"]) | set(recent["AwayTeam"])) - fitted
    if dropped:
        print(f"[warn] no DC rating, skipping: {sorted(dropped)}")
    print(f"[data] {len(teams)} teams with ratings")
    return teams

# ---- DC replacement for match_stats: returns same fields the prompt needs ----
def dc_match_stats(params, home, away):
    m = dixon_coles_simulate_match(params, home, away, max_goals=10)
    probs = get_1x2_probs(m)                      # {'H','D','A'}
    # expected goals = mean of each Poisson margin
    xg_h = float(np.dot(np.arange(m.shape[0]), m.sum(axis=1)))
    xg_a = float(np.dot(np.arange(m.shape[1]), m.sum(axis=0)))
    return {"xg_h": round(xg_h, 2), "xg_a": round(xg_a, 2),
            "p_h": round(probs["H"]*100, 1), "p_d": round(probs["D"]*100, 1),
            "p_a": round(probs["A"]*100, 1)}

# ============================================================================
# PREDICT ONE MATCH  (LLM decides, DC feeds numbers)
# ============================================================================
def predecir_partido(params, home, away, rng):
    s = dc_match_stats(params, home, away)
    system_prompt = (
        "Eres un analista experto de fútbol. Se te da UN partido de liga con "
        "estadísticas Dixon-Coles ya calculadas. Devuelve SOLO el marcador en "
        "formato 'Local X-X Visitante'. El empate SÍ está permitido. Una línea.")
    user_prompt = (
        f"Partido: {home} (local) vs {away} (visitante)\n"
        f"Goles esperados (Dixon-Coles): {home}={s['xg_h']}, {away}={s['xg_a']}\n"
        f"Prob. previa: {home}={s['p_h']}%, empate={s['p_d']}%, {away}={s['p_a']}%\n\n"
        "Marcador:")
    resp = llamar_local(system_prompt, user_prompt, f"{home} vs {away}")

    gh = ga = None
    if resp:
        mt = re.search(r"(\d+)\s*-\s*(\d+)", resp)
        if mt: gh, ga = int(mt.group(1)), int(mt.group(2))
    if gh is None:                                 # fallback: DC expected goals
        gh, ga = round(s["xg_h"]), round(s["xg_a"])
    return gh, ga

# ============================================================================
# FULL SEASON  (unchanged logic)
# ============================================================================
def simular_temporada(params, teams, rng, seed=42):
    tabla = {t: {"team": t, "PJ":0,"G":0,"E":0,"P":0,"GF":0,"GC":0,"Pts":0} for t in teams}
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    print(f"\n[liga] {len(fixtures)} partidos...")
    for k, (h, a) in enumerate(fixtures, 1):
        gh, ga = predecir_partido(params, h, a, rng)
        th, ta = tabla[h], tabla[a]
        th["PJ"]+=1; ta["PJ"]+=1; th["GF"]+=gh; th["GC"]+=ga; ta["GF"]+=ga; ta["GC"]+=gh
        if gh > ga: th["G"]+=1; ta["P"]+=1; th["Pts"]+=3
        elif ga > gh: ta["G"]+=1; th["P"]+=1; ta["Pts"]+=3
        else: th["E"]+=1; ta["E"]+=1; th["Pts"]+=1; ta["Pts"]+=1
        if k % 50 == 0 or k == len(fixtures): print(f"    {k}/{len(fixtures)}")
    t = pd.DataFrame(tabla.values())
    t["DG"] = t["GF"] - t["GC"]
    return t.sort_values(["Pts","DG","GF"], ascending=False).reset_index(drop=True)

# ============================================================================
# SAVE ACTIVATIONS (unchanged)
# ============================================================================
def guardar_activaciones():
    if not (SAVE_ACT and ACT_VECS): return
    M = np.stack(ACT_VECS, axis=1)
    np.save(os.path.join(OUT_DIR, "activations.npy"), M.T)
    with open(os.path.join(OUT_DIR, "activations_meta_liga.json"), "w") as f:
        json.dump({"labels": ACT_LABELS, "l_star": ACT_LSTAR}, f, indent=2)
    print("[act] saved activations.npy + activations_meta_liga.json")

# ============================================================================
# MAIN
# ============================================================================
def main():
    df = cargar_dataset()
    print("[dc] fitting Dixon-Coles ...")
    params = solve_parameters_decay(df, xi=XI, options={"disp": False, "maxiter": 100})
    teams = elegir_equipos(df, params)
    rng = np.random.default_rng(42)
    tabla = simular_temporada(params, teams, rng)
    print("\n===== CLASIFICACIÓN FINAL =====")
    print(tabla[["team","PJ","G","E","P","GF","GC","DG","Pts"]].to_string(index=False))
    print(f"\n🏆 CAMPEÓN 2027: {tabla.iloc[0]['team']} ({int(tabla.iloc[0]['Pts'])} pts)")
    tabla.to_csv(os.path.join(OUT_DIR, "standings_2027.csv"), index=False)
    guardar_activaciones()

if __name__ == "__main__":
    main()
