"""
app.py – Dashboard Interactivo · Copa Mundial 2026
===================================================
Interfaz Streamlit dark-mode corregida.
Modelos: Dixon-Coles + Montecarlo 100K.
"""

from __future__ import annotations

import warnings
import os
import time
import json
import threading
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA Y DATOS BASE
# ══════════════════════════════════════════════════════════════════════════════

REAL_RESULTS_FILE = "real_results.json"
_RESULTS_LOCK = threading.Lock()

def load_real_results() -> dict:
    if os.path.exists(REAL_RESULTS_FILE):
        try:
            with open(REAL_RESULTS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_real_results(res: dict):
    with _RESULTS_LOCK:
        with open(REAL_RESULTS_FILE, "w") as f:
            json.dump(res, f, indent=2)

NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "Turkey": "Türkiye", "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czechia", "DR Congo": "Congo DR", "Korea, South": "South Korea"
}

@st.cache_data(show_spinner=False)
def load_h2h_data():
    try:
        df = pd.read_csv("h2h_results.csv", parse_dates=["date"])
        df["home_team"] = df["home_team"].replace(NAME_MAP)
        df["away_team"] = df["away_team"].replace(NAME_MAP)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])

@st.cache_data(show_spinner=False)
def get_all_h2h_stats() -> dict:
    df = load_h2h_data()
    if df.empty: return {}
    stats = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if pd.isna(h) or pd.isna(a): continue
        
        is_friendly = row.get('tournament') == 'Friendly'
        weight = 0.1 if is_friendly else 1.0
        
        for t1, t2 in [(h, a), (a, h)]:
            if t1 not in stats: stats[t1] = {}
            if t2 not in stats[t1]: stats[t1][t2] = {"wins": 0.0, "draws": 0.0, "losses": 0.0, "matches": 0.0}
        
        stats[h][a]["matches"] += weight
        stats[a][h]["matches"] += weight
        if row["home_score"] > row["away_score"]:
            stats[h][a]["wins"] += weight
            stats[a][h]["losses"] += weight
        elif row["home_score"] < row["away_score"]:
            stats[h][a]["losses"] += weight
            stats[a][h]["wins"] += weight
        else:
            stats[h][a]["draws"] += weight
            stats[a][h]["draws"] += weight
    return stats

def get_h2h_multiplier(home: str, away: str, h2h_dict: dict) -> tuple[float, float]:
    if home not in h2h_dict or away not in h2h_dict[home]:
        return 1.0, 1.0
    rec = h2h_dict[home][away]
    if rec["matches"] < 3:
        return 1.0, 1.0
    wr = rec["wins"] / rec["matches"]
    lr = rec["losses"] / rec["matches"]
    return 1.0 + (wr - 0.5)*0.1, 1.0 + (lr - 0.5)*0.1

SIMULATION_RUNS = 100000
PRECISION_MODE = True
HOME_ADVANTAGE_XG = {"Mexico": 0.35, "United States": 0.20, "Canada": 0.15}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="WC2026 Predictor", page_icon="🏆", layout="wide")

st.markdown("""
<style>
    :root { --bg-primary: #0d1117; --bg-card: #1c2128; --text-primary: #e6edf3; --accent-blue: #58a6ff; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
    .stat-card { background: var(--bg-card); border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 10px; }
    .pill { padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .pill-blue { background: rgba(88,166,255,0.2); color: #58a6ff; }
    .pill-orange { background: rgba(240,136,62,0.2); color: #f0883e; }
    .score-badge { background: #1f6feb; border-radius: 8px; padding: 4px 12px; font-weight: 700; color: white; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA CANON
# ══════════════════════════════════════════════════════════════════════════════

WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "Congo DR"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_FLAGS = {
    "Argentina": "🇦🇷", "France": "🇫🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Brazil": "🇧🇷", "Portugal": "🇵🇹",
    "Germany": "🇩🇪", "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Japan": "🇯🇵", "Morocco": "🇲🇦", "United States": "🇺🇸", "Switzerland": "🇨🇭", "Sweden": "🇸🇪", "Mexico": "🇲🇽",
    "Türkiye": "🇹🇷", "Ecuador": "🇪🇨", "South Korea": "🇰🇷", "Senegal": "🇸🇳", "Australia": "🇦🇺", "Canada": "🇨🇦",
    "Norway": "🇳🇴", "Czechia": "🇨🇿", "South Africa": "🇿🇦", "Egypt": "🇪🇬", "Nigeria": "🇳🇬", "Paraguay": "🇵🇾",
    "Algeria": "🇩🇿", "Iran": "🇮🇷", "Ivory Coast": "🇨🇮", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Panama": "🇵🇦",
    "Uzbekistan": "🇺🇿", "New Zealand": "🇳🇿", "Jordan": "🇯🇴", "Ghana": "🇬🇭", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Congo DR": "🇨🇩"
}

QUALIFIED_TEAMS = [t for group in WC2026_GROUPS.values() for t in group]
CO_HOSTS = {"United States", "Mexico", "Canada"}
N_TEAMS = len(QUALIFIED_TEAMS)
TEAM_TO_IDX = {t: i for i, t in enumerate(QUALIFIED_TEAMS)}

_BASE_ELO = {
    "Argentina": 2100, "France": 2085, "Spain": 2075, "England": 2060, "Brazil": 2050,
    "Portugal": 2045, "Germany": 2035, "Netherlands": 2020, "Belgium": 2010, "Croatia": 1995,
    "Uruguay": 2000, "Colombia": 1990, "Japan": 1970, "Morocco": 1965, "United States": 1960,
    "Mexico": 1940, "Canada": 1900, "Italy": 1950, "Norway": 1940, "South Korea": 1920
}

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _elo_xg(elo_h: float, elo_a: float, home: str = "", away: str = "") -> tuple[float, float]:
    dr = elo_h - elo_a
    ea = 1 / (1 + 10 ** (-dr / 400))
    xg_h = ea * 2.7
    xg_a = (1 - ea) * 2.7
    xg_h += HOME_ADVANTAGE_XG.get(home, 0.0)
    xg_a += HOME_ADVANTAGE_XG.get(away, 0.0)
    return max(0.2, xg_h), max(0.2, xg_a)

@st.cache_data(show_spinner=False)
def calculate_form_multipliers() -> dict:
    return {t: 1.0 for t in QUALIFIED_TEAMS}

def _build_elo_xg_matrix() -> np.ndarray:
    xg = np.zeros((N_TEAMS, N_TEAMS, 2), dtype=np.float32)
    h2h = get_all_h2h_stats()
    form = calculate_form_multipliers()
    for i in range(N_TEAMS):
        for j in range(N_TEAMS):
            if i == j: continue
            ti, tj = QUALIFIED_TEAMS[i], QUALIFIED_TEAMS[j]
            xi, xj = _elo_xg(_BASE_ELO.get(ti, 1850), _BASE_ELO.get(tj, 1850), ti, tj)
            mi, mj = get_h2h_multiplier(ti, tj, h2h)
            xg[i, j, 0] = xi * mi * form.get(ti, 1.0)
            xg[i, j, 1] = xj * mj * form.get(tj, 1.0)
    return xg

# ══════════════════════════════════════════════════════════════════════════════
# SIMULACIÓN (ESTA SECCIÓN CONTIENE EL FIX)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_simulation_results(_real_res_hashable: tuple):
    n = SIMULATION_RUNS
    real_res = dict(_real_res_hashable)
    idx_to_team = {v: k for k, v in TEAM_TO_IDX.items()}
    xg_mat = _build_elo_xg_matrix()
    rng = np.random.default_rng(2026)

    # Acumuladores
    survival = np.zeros((N_TEAMS, 7), dtype=np.int32)
    survival[:, 0] = n
    pts = np.zeros((n, N_TEAMS), dtype=np.int16)
    gd = np.zeros((n, N_TEAMS), dtype=np.int16)
    gf = np.zeros((n, N_TEAMS), dtype=np.int16)

    # 1. Fase de Grupos
    for letter, teams in WC2026_GROUPS.items():
        indices = [TEAM_TO_IDX[t] for t in teams]
        for i, j in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
            hi, ai = indices[i], indices[j]
            th, ta = QUALIFIED_TEAMS[hi], QUALIFIED_TEAMS[ai]
            key = f"{th}|{ta}"
            if key in real_res:
                gh = np.full(n, real_res[key]["g_h"], dtype=np.int16)
                ga = np.full(n, real_res[key]["g_a"], dtype=np.int16)
            else:
                gh = rng.poisson(xg_mat[hi, ai, 0], n).astype(np.int16)
                ga = rng.poisson(xg_mat[hi, ai, 1], n).astype(np.int16)
            
            pts[:, hi] += np.where(gh > ga, 3, np.where(gh == ga, 1, 0))
            pts[:, ai] += np.where(ga > gh, 3, np.where(gh == ga, 1, 0))
            gd[:, hi] += (gh - ga); gd[:, ai] += (ga - gh)
            gf[:, hi] += gh; gf[:, ai] += ga

    # Clasificación
    score = pts.astype(np.float32) * 1000 + gd.astype(np.float32) + gf.astype(np.float32) * 0.01 + rng.random((n, N_TEAMS)) * 0.001
    
    winners = np.zeros((n, 12), dtype=np.int32)
    runners = np.zeros((n, 12), dtype=np.int32)
    thirds = np.zeros((n, 12), dtype=np.int32)

    for g_idx, (letter, teams) in enumerate(WC2026_GROUPS.items()):
        indices = np.array([TEAM_TO_IDX[t] for t in teams])
        group_scores = score[:, indices]
        ranks = np.argsort(-group_scores, axis=1)
        
        # FIX: Indexación avanzada en lugar de take_along_axis
        winners[:, g_idx] = indices[ranks[:, 0]]
        runners[:, g_idx] = indices[ranks[:, 1]]
        thirds[:, g_idx]  = indices[ranks[:, 2]]

    # Mejores terceros
    t_scores = np.take_along_axis(score, thirds, axis=1)
    t_ranks = np.argsort(-t_scores, axis=1)[:, :8]
    best_thirds = np.take_along_axis(thirds, t_ranks, axis=1)

    # 2. Bracket R32
    # Combinamos Ganadores, Segundos y 8 Mejores Terceros
    r32_h = np.concatenate([winners, best_thirds[:, [0, 2, 4, 6]]], axis=1)
    idx_inv = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    r32_a = np.concatenate([runners[:, idx_inv], best_thirds[:, [1, 3, 5, 7]]], axis=1)

    def simulate_ko(h_mat, a_mat, phase_idx):
        num_m = h_mat.shape[1]
        next_w = np.zeros((n, num_m), dtype=np.int32)
        for m in range(num_m):
            h, a = h_mat[:, m], a_mat[:, m]
            gh = rng.poisson(xg_mat[h, a, 0], n)
            ga = rng.poisson(xg_mat[h, a, 1], n)
            pk = rng.random(n) < 0.5
            winner = np.where(gh > ga, h, np.where(ga > gh, a, np.where(pk, h, a)))
            next_w[:, m] = winner
            u, c = np.unique(winner, return_counts=True)
            survival[u, phase_idx] += c
        return next_w

    # Ejecución
    u32, c32 = np.unique(np.concatenate([r32_h, r32_a]), return_counts=True)
    survival[u32, 1] += c32
    r16 = simulate_ko(r32_h, r32_a, 2)
    qf = simulate_ko(r16[:, :8], r16[:, 8:], 3)
    sf = simulate_ko(qf[:, :4], qf[:, 4:], 4)
    fin = simulate_ko(sf[:, :2], sf[:, 2:], 5)
    simulate_ko(fin[:, :1], fin[:, 1:], 6)

    df_res = pd.DataFrame({"Team": QUALIFIED_TEAMS})
    cols = ["Fase Grupos %", "R32 %", "R16 %", "Cuartos %", "Semis %", "Final %", "Ganador %"]
    for i, col in enumerate(cols):
        df_res[col] = (survival[:, i] / n) * 100

    return df_res.sort_values("Ganador %", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏆 WC2026 AI Predictor")
tabs = st.tabs(["📊 Simulación", "📝 Resultados en Vivo", "🌐 Power Ranking"])

with tabs[0]:
    if st.button("⚡ Ejecutar 100,000 Simulaciones"):
        real_res = load_real_results()
        df = get_simulation_results(tuple(sorted(real_res.items())))
        st.dataframe(df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.info("Ingresa resultados reales para ajustar las proyecciones.")
    with st.form("res_form"):
        c1, c2, c3, c4 = st.columns([3,1,1,3])
        t_h = c1.selectbox("Local", QUALIFIED_TEAMS)
        g_h = c2.number_input("G", 0, 15)
        g_a = c3.number_input("G ", 0, 15)
        t_a = c4.selectbox("Visita", [t for t in QUALIFIED_TEAMS if t != t_h])
        if st.form_submit_button("Guardar"):
            res = load_real_results()
            res[f"{t_h}|{t_a}"] = {"g_h": g_h, "g_a": g_a}
            save_real_results(res)
            st.rerun()

with tabs[2]:
    st.markdown("### Elo Rating FIFA 2026")
    elo_df = pd.DataFrame([{"Equipo": t, "Elo": _BASE_ELO.get(t, 1850)} for t in QUALIFIED_TEAMS])
    st.table(elo_df.sort_values("Elo", ascending=False).head(15))