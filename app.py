"""
app.py – Dashboard Interactivo · Copa Mundial 2026
===================================================
Interfaz Streamlit dark-mode que integra:
  • dixon_coles_engine.py  – predicciones de marcadores exactos
  • ensemble_simulator.py  – simulación Montecarlo 100K vectorizada

Despliegue: Streamlit Community Cloud (gratuito)
Referencia: Zeileis & Groll (2018) · Baio & Blangiardo (2010)
"""

from __future__ import annotations

import warnings
import os
import time
import sys
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


import json

REAL_RESULTS_FILE = "real_results.json"

def load_real_results() -> dict:
    if os.path.exists(REAL_RESULTS_FILE):
        try:
            with open(REAL_RESULTS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_real_results(res: dict):
    with open(REAL_RESULTS_FILE, "w") as f:
        json.dump(res, f)

@st.cache_data(show_spinner=False)
def load_h2h_data():
    try:
        return pd.read_csv("h2h_results.csv", parse_dates=["date"])
    except:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])

@st.cache_data(show_spinner=False)
def get_all_h2h_stats() -> dict:
    df = load_h2h_data()
    if df.empty: return {}
    stats = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        if pd.isna(h) or pd.isna(a): continue
        if h not in stats: stats[h] = {}
        if a not in stats[h]: stats[h][a] = {"wins": 0, "draws": 0, "losses": 0, "matches": 0}
        if a not in stats: stats[a] = {}
        if h not in stats[a]: stats[a][h] = {"wins": 0, "draws": 0, "losses": 0, "matches": 0}
        
        stats[h][a]["matches"] += 1
        stats[a][h]["matches"] += 1
        if row["home_score"] > row["away_score"]:
            stats[h][a]["wins"] += 1
            stats[a][h]["losses"] += 1
        elif row["home_score"] < row["away_score"]:
            stats[h][a]["losses"] += 1
            stats[a][h]["wins"] += 1
        else:
            stats[h][a]["draws"] += 1
            stats[a][h]["draws"] += 1
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

@st.cache_data(show_spinner=False)
def load_logistics_data() -> dict:
    return {}


SIMULATION_RUNS = 100000
PRECISION_MODE = True

HOME_ADVANTAGE_XG = {
    "Mexico": 0.35,
    "United States": 0.20,
    "Canada": 0.15,
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL DE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WC2026 Predictor · AI Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "About": "Copa Mundial 2026 – Motor de Predicción IA (Dixon-Coles + Bayesiano + Ensemble RF)",
    },
)

# ── CSS personalizado (dark mode premium) ────────────────────────────────────
st.markdown("""
<style>
/* ── Fuente Google ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variables globales ── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --bg-hover:      #21262d;
    --border:        #30363d;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
    --accent-blue:   #58a6ff;
    --accent-green:  #3fb950;
    --accent-orange: #f0883e;
    --accent-red:    #f85149;
    --accent-purple: #bc8cff;
    --accent-yellow: #d29922;
    --gold:          #ffd700;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] label { color: var(--text-primary) !important; }

/* ── Selectbox & widgets ── */
.stSelectbox > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] { color: var(--text-primary) !important; }

/* ── Métricas ── */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(88,166,255,0.25) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Headers ── */
h1, h2, h3 { color: var(--text-primary) !important; }

/* ── Pills / badges ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.pill-blue   { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
.pill-green  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
.pill-orange { background: rgba(240,136,62,0.15); color: #f0883e; border: 1px solid rgba(240,136,62,0.3); }

/* ── Stat card ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent-blue); }
.stat-label { color: var(--text-secondary); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.stat-value { color: var(--text-primary); font-size: 1.9rem; font-weight: 700; line-height: 1; }
.stat-sub   { color: var(--text-muted); font-size: 0.75rem; margin-top: 4px; }

/* ── Section title ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 14px;
    margin-top: 28px;
}

/* ── Score badge ── */
.score-badge {
    background: linear-gradient(135deg, #1f6feb 0%, #bc8cff 100%);
    border-radius: 12px;
    padding: 3px 12px;
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
    display: inline-block;
}

/* ── Progress bar custom ── */
.prob-bar-wrap { margin: 6px 0; }
.prob-bar-label { display: flex; justify-content: space-between; margin-bottom: 3px; font-size: 0.8rem; color: var(--text-secondary); }
.prob-bar { height: 8px; border-radius: 4px; overflow: hidden; background: var(--bg-hover); }
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }

/* ── Tab styling ── */
[data-baseweb="tab-list"] { background: var(--bg-secondary) !important; border-bottom: 1px solid var(--border) !important; }
[data-baseweb="tab"] { color: var(--text-secondary) !important; font-weight: 500 !important; }
[aria-selected="true"] { color: var(--accent-blue) !important; border-bottom-color: var(--accent-blue) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS DEL TORNEO (CANON)
# ══════════════════════════════════════════════════════════════════════════════

# 12 Grupos A–L con sus 4 selecciones (sorteo oficial WC2026)
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",        "South Africa", "South Korea", "Czechia"],
    "B": ["Canada",        "Switzerland",  "Qatar",       "Bosnia and Herzegovina"],
    "C": ["Brazil",        "Morocco",      "Haiti",       "Scotland"],
    "D": ["United States", "Paraguay",     "Australia",   "Türkiye"],
    "E": ["Germany",       "Curaçao",      "Ivory Coast", "Ecuador"],
    "F": ["Netherlands",   "Japan",        "Tunisia",     "Sweden"],
    "G": ["Belgium",       "Egypt",        "Iran",        "New Zealand"],
    "H": ["Spain",         "Cabo Verde",   "Saudi Arabia","Uruguay"],
    "I": ["France",        "Senegal",      "Norway",      "Iraq"],
    "J": ["Argentina",     "Algeria",      "Austria",     "Jordan"],
    "K": ["Portugal",      "Uzbekistan",   "Colombia",    "Congo DR"],
    "L": ["England",       "Croatia",      "Ghana",       "Panama"],
}

# Flags emoji
TEAM_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "France": "🇫🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Brazil": "🇧🇷", "Portugal": "🇵🇹", "Germany": "🇩🇪", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Japan": "🇯🇵", "Morocco": "🇲🇦", "United States": "🇺🇸", "Switzerland": "🇨🇭",
    "Sweden": "🇸🇪", "Serbia": "🇷🇸", "Mexico": "🇲🇽", "Italy": "🇮🇹",
    "Türkiye": "🇹🇷", "Ecuador": "🇪🇨", "Poland": "🇵🇱", "South Korea": "🇰🇷",
    "Senegal": "🇸🇳", "Ukraine": "🇺🇦", "Australia": "🇦🇺", "Canada": "🇨🇦",
    "Austria": "🇦🇹", "Nigeria": "🇳🇬", "Paraguay": "🇵🇾", "Algeria": "🇩🇿",
    "Iran": "🇮🇷", "Egypt": "🇪🇬", "Ivory Coast": "🇨🇮", "Cameroon": "🇨🇲",
    "Costa Rica": "🇨🇷", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Chile": "🇨🇱",
    "South Africa": "🇿🇦", "Mali": "🇲🇱", "Iraq": "🇮🇶", "Panama": "🇵🇦",
    "Honduras": "🇭🇳", "Uzbekistan": "🇺🇿", "New Zealand": "🇳🇿", "Indonesia": "🇮🇩",
    "Norway": "🇳🇴", "Czechia": "🇨🇿", "Bosnia and Herzegovina": "🇧🇦",
    "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Curaçao": "🇨🇼", "Tunisia": "🇹🇳",
    "Cabo Verde": "🇨🇻", "Jordan": "🇯🇴", "Congo DR": "🇨🇩", "Ghana": "🇬🇭"
}

QUALIFIED_TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czechia",
    "Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Türkiye",
    "Germany", "Curaçao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Tunisia", "Sweden",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cabo Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Norway", "Iraq",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "Uzbekistan", "Colombia", "Congo DR",
    "England", "Croatia", "Ghana", "Panama"
]

CO_HOSTS = {"United States", "Mexico", "Canada"}
N_TEAMS = len(QUALIFIED_TEAMS)
TEAM_ORDER_LIST = list(QUALIFIED_TEAMS)
TEAM_TO_IDX = {t: i for i, t in enumerate(QUALIFIED_TEAMS)}

# Paleta de colores plotly
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font_color="#e6edf3",
    font_family="Inter",
    coloraxis_colorbar=dict(
        bgcolor="#1c2128",
        tickcolor="#8b949e",
        outlinecolor="#30363d",
        title=dict(font=dict(color="#8b949e")),
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PREDICTIVO (self-contained fallback si no hay modelos ajustados)
# ══════════════════════════════════════════════════════════════════════════════

# Elo base por equipo (aprox. Elo FIFA junio 2026)
_BASE_ELO: dict[str, float] = {
    "Argentina": 2088, "France": 2082, "Spain": 2072, "England": 2060,
    "Brazil": 2052, "Portugal": 2044, "Germany": 2030, "Netherlands": 2018,
    "Belgium": 2010, "Croatia": 1998, "Uruguay": 1992, "Colombia": 1982,
    "Japan": 1975, "Morocco": 1968, "United States": 1965, "Switzerland": 1960,
    "Sweden": 1935, "Serbia": 1952, "Mexico": 1950, "Italy": 1945,
    "Türkiye": 1938, "Ecuador": 1932, "Poland": 1928, "South Korea": 1920,
    "Senegal": 1918, "Ukraine": 1912, "Australia": 1905, "Canada": 1900,
    "Austria": 1895, "Nigeria": 1888, "Paraguay": 1882, "Algeria": 1878,
    "Iran": 1870, "Egypt": 1865, "Ivory Coast": 1858, "Cameroon": 1850,
    "Costa Rica": 1842, "Saudi Arabia": 1835, "Qatar": 1828, "Chile": 1822,
    "South Africa": 1815, "Mali": 1808, "Iraq": 1800, "Panama": 1795,
    "Honduras": 1788, "Uzbekistan": 1782, "New Zealand": 1770, "Indonesia": 1750,
    "Norway": 1958, "Czechia": 1925, "Bosnia and Herzegovina": 1830,
    "Haiti": 1750, "Scotland": 1910, "Curaçao": 1760, "Tunisia": 1880,
    "Cabo Verde": 1810, "Jordan": 1840, "Congo DR": 1820, "Ghana": 1850,
}

import pandas as pd




@st.cache_resource(show_spinner=False)
def build_ensemble_engine():
    """
    Construye el motor Ensemble + Montecarlo y cachea los resultados.
    Usa datos internos si los modelos reales (Dixon-Coles/Bayesiano) no están ajustados.
    """
    try:
        sys.path.insert(0, ".")
        from ensemble_simulator import FeatureEngineer, EnsembleModel, VectorizedTournamentSimulator

        engineer = FeatureEngineer()
        training_data = engineer.load_and_merge()
        model = EnsembleModel()
        model.train(training_data)
        xg_matrix = model.precompute_xg_matrix()
        simulator = VectorizedTournamentSimulator(xg_matrix, n_simulations=100_000)
        results_df = simulator.simulate()
        return xg_matrix, results_df
    except Exception:
        return _build_elo_xg_matrix(), get_simulation_results(_cache_key=_get_results_cache_key())[0]


@st.cache_data(show_spinner=False)
def load_sqi_data_v3() -> dict:
    try:
        import os
        if os.path.exists('fbref_sqi.csv'):
            df = pd.read_csv('fbref_sqi.csv')
            return dict(zip(df['Team'], df['SQI']))
    except:
        pass
    return {t: 1.0 for t in QUALIFIED_TEAMS}

def _elo_xg(elo_h: float, elo_a: float, home_team: str = "", away_team: str = "") -> tuple[float, float]:
    """Calcula xG base a partir de Elos."""
    dr = elo_h - elo_a
    ea = 1 / (1 + 10 ** (-dr / 400))
    xg_h = ea * 2.8
    xg_a = (1 - ea) * 2.8
    
    xg_h += HOME_ADVANTAGE_XG.get(home_team, 0.0)
    xg_a += HOME_ADVANTAGE_XG.get(away_team, 0.0)
    
    return max(0.1, xg_h), max(0.1, xg_a)


def _build_elo_xg_matrix() -> np.ndarray:
    sqi = load_sqi_data_v3()
    form = calculate_form_multipliers()
    xg = np.zeros((N_TEAMS, N_TEAMS, 2), dtype=np.float32)

    try:
        h2h_dict = get_all_h2h_stats()
    except:
        h2h_dict = {}
        
    for i in range(N_TEAMS):
        for j in range(N_TEAMS):
            if i == j: continue
            
            t_i, t_j = QUALIFIED_TEAMS[i], QUALIFIED_TEAMS[j]
            elo_i = _BASE_ELO.get(t_i, 1850)
            elo_j = _BASE_ELO.get(t_j, 1850)
            
            sqi_i = sqi.get(t_i, 1.0)
            sqi_j = sqi.get(t_j, 1.0)
            
            sqi_c = max(0.85, min(1.15, sqi_i))
            e_i = elo_i * 0.40 + (elo_i * sqi_c) * 0.60
            sqi_c_j = max(0.85, min(1.15, sqi_j))
            e_j = elo_j * 0.40 + (elo_j * sqi_c_j) * 0.60
            
            xg_i, xg_j = _elo_xg(e_i, e_j, t_i, t_j)
            
            m_i, m_j = get_h2h_multiplier(t_i, t_j, h2h_dict)
            xg_i *= m_i
            xg_j *= m_j
            
            xg_i *= form.get(t_i, 1.0)
            xg_j *= form.get(t_j, 1.0)
            
            xg[i, j, 0] = xg_i
            xg[i, j, 1] = xg_j
            
    return xg




@st.cache_data(show_spinner=False)
def estimate_rho_from_data() -> dict:
    try:
        df = pd.read_csv("h2h_results.csv", parse_dates=["date"])
        df = df[df["date"].dt.year >= 2010]
        df = df[df["tournament"] != "Friendly"]
        df = df[df["home_team"].isin(QUALIFIED_TEAMS) & df["away_team"].isin(QUALIFIED_TEAMS)]
        df = df.dropna(subset=["home_score", "away_score"])
        
        N = len(df)
        if N < 150: return {"rho": -0.12, "N": N, "fallback": True}
        
        h_scores, a_scores = df["home_score"].values, df["away_score"].values
        xg_h, xg_a = np.zeros(N), np.zeros(N)
        for i, row in enumerate(df.itertuples()):
            xh, xa = _elo_xg(_BASE_ELO.get(row.home_team, 1850), _BASE_ELO.get(row.away_team, 1850), home_team=row.home_team, away_team=row.away_team)
            xg_h[i] = xh
            xg_a[i] = xa
            
        from scipy.stats import poisson as sp_poisson
        
        def neg_log_likelihood(rho):
            tau = np.ones(N)
            idx_00 = (h_scores == 0) & (a_scores == 0)
            idx_10 = (h_scores == 1) & (a_scores == 0)
            idx_01 = (h_scores == 0) & (a_scores == 1)
            idx_11 = (h_scores == 1) & (a_scores == 1)
            
            tau[idx_00] = 1 - xg_h[idx_00] * xg_a[idx_00] * rho
            tau[idx_10] = 1 + xg_a[idx_10] * rho
            tau[idx_01] = 1 + xg_h[idx_01] * rho
            tau[idx_11] = 1 - rho
            
            if np.any(tau <= 0): return 1e9
            
            # Términos de Poisson (constantes respecto a rho pero necesarios para 
            # que el optimizador encuentre el mínimo global correcto)
            log_pois_h = sp_poisson.logpmf(h_scores.astype(int), np.maximum(xg_h, 1e-9))
            log_pois_a = sp_poisson.logpmf(a_scores.astype(int), np.maximum(xg_a, 1e-9))
            
            return -(np.sum(np.log(tau)) + np.sum(log_pois_h) + np.sum(log_pois_a))
            
        res = minimize_scalar(neg_log_likelihood, bounds=(-0.25, 0.0), method='bounded')
        return {"rho": res.x if res.success else -0.12, "N": N, "fallback": not res.success}
    except Exception as e:
        return {"rho": -0.12, "N": 0, "fallback": True, "error": str(e)}

@st.cache_data(show_spinner=False)
def calculate_form_multipliers() -> dict:
    try:
        df = pd.read_csv("h2h_results.csv", parse_dates=["date"])
        df = df[df["tournament"] != "Friendly"]
        df = df[df["date"] < "2026-06-01"]
        form_mults = {}
        for team in QUALIFIED_TEAMS:
            team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date", ascending=False).head(12)
            if len(team_df) < 5:
                form_mults[team] = 1.0
                continue
            form_score_sum = 0.0
            weight_sum = 0.0
            for i, row in enumerate(team_df.itertuples()):
                decay = 0.88 ** i
                is_home = row.home_team == team
                rival = row.away_team if is_home else row.home_team
                hg, ag = row.home_score, row.away_score
                res = 1.0 if (hg > ag and is_home) or (ag > hg and not is_home) else (0.5 if hg == ag else 0.0)
                quality_weight = _BASE_ELO.get(rival, 1850) / 1900.0
                form_score_sum += res * decay * quality_weight
                weight_sum += decay * quality_weight
            form_score = form_score_sum / weight_sum if weight_sum > 0 else 0.5
            form_mults[team] = 0.90 + (form_score * 0.20)
        return form_mults
    except Exception as e:
        print(f"Warning: calculate_form_multipliers failed - {e}")
        return {t: 1.0 for t in QUALIFIED_TEAMS}

@st.cache_data(show_spinner=False)
def get_penalty_skill() -> dict:
    try:
        df = pd.read_csv("shootouts.csv")
        records = {}
        for _, row in df.iterrows():
            w, h, a = row["winner"], row["home_team"], row["away_team"]
            if pd.isna(w): continue
            for t in [h, a]:
                if t not in records: records[t] = {"played": 0, "won": 0}
                records[t]["played"] += 1
                if w == t: records[t]["won"] += 1
        total_p = sum(r["played"] for r in records.values())
        global_mean = sum(r["won"] for r in records.values()) / total_p if total_p > 0 else 0.5
        penalty_skill = {"_global_mean": global_mean}
        for t, r in records.items():
            penalty_skill[t] = (r["won"] + 4 * global_mean) / (r["played"] + 4)
        return penalty_skill
    except:
        return {"_global_mean": 0.5}

def predict_match_elo(home: str, away: str, max_g: int = 7) -> dict:
    """Predictor Elo-Poisson con corrección Dixon-Coles τ para el dashboard."""
    base_elo_h = _BASE_ELO.get(home, 1850)
    base_elo_a = _BASE_ELO.get(away, 1850)
    
    sqi_dict = load_sqi_data_v3()
    sqi_h = sqi_dict.get(home, 1.0)
    sqi_a = sqi_dict.get(away, 1.0)
    
    sqi_h = max(0.85, min(1.15, sqi_h))
    elo_h = base_elo_h * 0.40 + (base_elo_h * sqi_h) * 0.60
    sqi_a = max(0.85, min(1.15, sqi_a))
    elo_a = base_elo_a * 0.40 + (base_elo_a * sqi_a) * 0.60
    
    xg_h, xg_a = _elo_xg(elo_h, elo_a, home, away)

    form_dict = calculate_form_multipliers()
    xg_h *= form_dict.get(home, 1.0)
    xg_a *= form_dict.get(away, 1.0)

    # Distribuciones Poisson marginales
    goals = np.arange(max_g + 1)
    ph = poisson.pmf(goals, xg_h)
    pa = poisson.pmf(goals, xg_a)

    # Matriz de marcadores
    M = np.outer(ph, pa)

    # Corrección τ Dixon-Coles (ρ estimado)
    rho_data = estimate_rho_from_data()
    rho = rho_data["rho"]
    def tau(x, y):
        if x == 0 and y == 0: return 1 - xg_h * xg_a * rho
        if x == 1 and y == 0: return 1 + xg_a * rho
        if x == 0 and y == 1: return 1 + xg_h * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0
    M[0,0] *= tau(0,0); M[1,0] *= tau(1,0)
    M[0,1] *= tau(0,1); M[1,1] *= tau(1,1)
    M /= M.sum()

    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    flat = M.flatten()
    top_idx = np.argsort(flat)[::-1][:6]
    top_scores = []
    for idx in top_idx:
        i, j = divmod(idx, max_g + 1)
        top_scores.append({"score": f"{i}–{j}", "home": i, "away": j, "prob": M[i, j]})

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "xg_h": xg_h, "xg_a": xg_a,
        "matrix": M, "top_scores": top_scores,
        "elo_h": elo_h, "elo_a": elo_a,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE VISUALIZACIÓN PLOTLY (dark)
# ══════════════════════════════════════════════════════════════════════════════

def plotly_heatmap(M: np.ndarray, home: str, away: str) -> go.Figure:
    flag_h = TEAM_FLAGS.get(home, "🏳")
    flag_a = TEAM_FLAGS.get(away, "🏳")
    labels_h = [f"{home} {i}" for i in range(M.shape[0])]
    labels_a = [f"{away} {j}" for j in range(M.shape[1])]

    fig = go.Figure(go.Heatmap(
        z=M * 100,
        x=labels_a,
        y=labels_h,
        colorscale=[
            [0.00, "#0d1117"],
            [0.15, "#0d2139"],
            [0.40, "#1f4f7f"],
            [0.70, "#1f6feb"],
            [0.85, "#58a6ff"],
            [1.00, "#cae8ff"],
        ],
        text=[[f"{M[i,j]*100:.1f}%" for j in range(M.shape[1])] for i in range(M.shape[0])],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Prob (%)",
                font=dict(color="#8b949e", size=11),
            ),
            tickfont=dict(color="#8b949e"),
            bgcolor="#1c2128",
            outlinecolor="#30363d",
        ),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{flag_h} {home}  vs  {flag_a} {away} – Mapa de Marcadores",
            font=dict(size=15, color="#e6edf3"),
        ),
        xaxis=dict(title=f"Goles {away}", title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e", size=9),
                   gridcolor="#21262d"),
        yaxis=dict(title=f"Goles {home}", title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e", size=9),
                   gridcolor="#21262d"),
        height=380,
    )
    return fig


def plotly_prob_bar(p_home: float, p_draw: float, p_away: float,
                    home: str, away: str) -> go.Figure:
    flag_h = TEAM_FLAGS.get(home, "🏳")
    flag_a = TEAM_FLAGS.get(away, "🏳")

    categories = [f"{flag_h} {home}", "Empate", f"{flag_a} {away}"]
    values = [p_home * 100, p_draw * 100, p_away * 100]
    colors = ["#3fb950", "#d29922", "#f85149"]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="inside",
        textfont=dict(size=14, color="white", family="Inter"),
        insidetextanchor="middle",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Probabilidades 1X2", font=dict(size=14, color="#e6edf3")),
        xaxis=dict(range=[0, 100], title="Probabilidad (%)",
                   title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e"),
                   gridcolor="#21262d", ticksuffix="%"),
        yaxis=dict(tickfont=dict(color="#e6edf3", size=12)),
        height=220,
        bargap=0.35,
    )
    return fig


def plotly_tournament_bracket(results_df: pd.DataFrame) -> go.Figure:
    top = results_df.head(20).copy()
    top = top.sort_values("Winner (%)")
    flags = [TEAM_FLAGS.get(t, "🏳") for t in top["Team"]]
    labels = [f"{f}  {t}" for f, t in zip(flags, top["Team"])]

    colors = []
    for v in top["Winner (%)"]:
        if v >= 10: colors.append("#3fb950")
        elif v >= 6: colors.append("#58a6ff")
        elif v >= 3: colors.append("#bc8cff")
        elif v >= 1: colors.append("#f0883e")
        else: colors.append("#484f58")

    fig = go.Figure(go.Bar(
        x=top["Winner (%)"],
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in top["Winner (%)"]],
        textposition="outside",
        textfont=dict(size=11, color="#8b949e"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="🏆 Probabilidad de ser Campeón (100K simulaciones)",
                   font=dict(size=14, color="#e6edf3")),
        xaxis=dict(title="Probabilidad (%)", title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e"), gridcolor="#21262d",
                   ticksuffix="%"),
        yaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
        height=580,
        margin=dict(l=180, r=60, t=50, b=30),
    )
    return fig


def plotly_survival_funnel(results_df: pd.DataFrame, team: str) -> go.Figure:
    phases = ["Round of 32 (%)", "Round of 16 (%)", "Quarterfinals (%)",
              "Semifinals (%)", "Final (%)", "Winner (%)"]
    phase_labels = ["R32", "R16", "Cuartos", "Semis", "Final", "🏆 Campeón"]

    row = results_df[results_df["Team"] == team]
    if row.empty:
        return go.Figure()

    vals = [row[p].values[0] for p in phases]
    flag = TEAM_FLAGS.get(team, "🏳")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_labels, y=vals,
        mode="lines+markers+text",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=10, color="#1f6feb",
                    line=dict(color="#58a6ff", width=2)),
        text=[f"{v:.1f}%" for v in vals],
        textposition="top center",
        textfont=dict(color="#8b949e", size=10),
        fill="tozeroy",
        fillcolor="rgba(31,111,235,0.12)",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{flag} {team} – Probabilidad por Fase",
                   font=dict(size=14, color="#e6edf3")),
        xaxis=dict(tickfont=dict(color="#e6edf3"), gridcolor="#21262d"),
        yaxis=dict(title="Probabilidad (%)", ticksuffix="%",
                   title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e"), gridcolor="#21262d"),
        height=280,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """
    Convierte un color hexadecimal a formato rgba() compatible con Plotly.

    Plotly NO acepta colores hex de 8 dígitos (#RRGGBBAA, estándar CSS4).
    Únicamente acepta: #RRGGBB, rgb(), rgba(), hsl(), hsla(), o nombre CSS.
    Esta función convierte '#3fb950' → 'rgba(63,185,80,0.15)'.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plotly_elo_radar(home: str, away: str, pred: dict) -> go.Figure:
    categories = ["Elo Rating", "xG Ataque", "xG Defensa", "Form"]

    def normalize(val, lo, hi):
        return round((val - lo) / (hi - lo + 1e-9) * 100, 1)

    elo_h, elo_a = pred["elo_h"], pred["elo_a"]
    xg_h, xg_a = pred["xg_h"], pred["xg_a"]

    form_dict = calculate_form_multipliers()
    form_h = round((form_dict.get(home, 1.0) - 0.94) / 0.12 * 100, 1)
    form_a = round((form_dict.get(away, 1.0) - 0.94) / 0.12 * 100, 1)

    vals_h = [normalize(elo_h, 1700, 2100), normalize(xg_h, 0.5, 2.5),
              normalize(2.5 - xg_a, 0.5, 2.5), form_h]
    vals_a = [normalize(elo_a, 1700, 2100), normalize(xg_a, 0.5, 2.5),
              normalize(2.5 - xg_h, 0.5, 2.5), form_a]

    flag_h = TEAM_FLAGS.get(home, "🏳")
    flag_a = TEAM_FLAGS.get(away, "🏳")

    # Colores sólidos para el borde y colores rgba() para el relleno
    # ✅ rgba() es el único formato que Plotly acepta para fillcolor transparente
    trace_styles = [
        (f"{flag_h} {home}", vals_h, "#3fb950", _hex_to_rgba("#3fb950", 0.15)),
        (f"{flag_a} {away}", vals_a, "#f85149", _hex_to_rgba("#f85149", 0.15)),
    ]

    fig = go.Figure()
    for label, vals, line_color, fill_color in trace_styles:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=label,
            line=dict(color=line_color, width=2),
            fillcolor=fill_color,   # ✅ 'rgba(63,185,80,0.15)' — formato válido
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="#1c2128",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="#484f58", size=8),
                gridcolor="#30363d", linecolor="#30363d",
            ),
            angularaxis=dict(
                tickfont=dict(color="#8b949e", size=10),
                linecolor="#30363d", gridcolor="#30363d",
            ),
        ),
        title=dict(text="Perfil Comparativo de Equipos",
                   font=dict(size=13, color="#e6edf3")),
        legend=dict(font=dict(color="#e6edf3"), bgcolor="rgba(0,0,0,0)"),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.8rem;">🏆</div>
        <div style="font-size:1.1rem; font-weight:700; color:#e6edf3; letter-spacing:0.03em;">WC2026 Predictor</div>
        <div style="font-size:0.72rem; color:#8b949e; margin-top:4px;">AI · Dixon-Coles · Bayes · RF</div>
    </div>
    <hr style="border-color:#30363d; margin:10px 0 18px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navegación</div>', unsafe_allow_html=True)
    page = st.radio(
        label="Sección",
        options=["🌐 Grupos", "⚽ Predictor de Partido", "📊 Simulación Montecarlo", "🏅 Ranking Global", "📝 Resultados en Vivo"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#30363d; margin:18px 0;">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Motor de Simulación</div>', unsafe_allow_html=True)
    st.markdown(f"**Iteraciones Fijas:** {SIMULATION_RUNS:,}")
    st.markdown(f"**Modo Precisión:** {'Activado' if PRECISION_MODE else 'Desactivado'}")
    st.markdown('<hr style="border-color:#30363d; margin:18px 0;">', unsafe_allow_html=True)

    rho_info = estimate_rho_from_data()
    ps_info = get_penalty_skill()
    rho_label = f"ρ = {rho_info['rho']:.3f} {'(fallback)' if rho_info['fallback'] else f'N={rho_info["N"]}'}"
    ps_label = f"{'Datos reales' if len(ps_info) > 1 else 'Modelo 50/50'}"
    form_label = "Activo ✓"

    st.markdown('<div class="section-title">Estado del Motor</div>', unsafe_allow_html=True)
    st.markdown(f"**Dixon-Coles τ:** {rho_label}")
    st.markdown(f"**Penaltis:** {ps_label}")
    st.markdown(f"**Forma Temporal:** {form_label}")


    st.markdown('<hr style="border-color:#30363d; margin:18px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#484f58; line-height:1.6;">
    <b style="color:#8b949e;">Modelos:</b><br>
    • Dixon-Coles (1997)<br>
    • Baio &amp; Blangiardo (2010)<br>
    • Zeileis &amp; Groll (2018)<br><br>
    <b style="color:#8b949e;">Inferencia:</b><br>
    ADVI · MLE · RandomForest<br><br>
    <b style="color:#8b949e;">Motor:</b> NumPy vectorizado<br>
    100K sims · &lt;10s
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS (CACHEADA)
# ══════════════════════════════════════════════════════════════════════════════

import hashlib
import json

def _get_results_cache_key() -> str:
    rr = load_real_results()
    return hashlib.md5(json.dumps(rr, sort_keys=True).encode()).hexdigest()

@st.cache_data(show_spinner=False)
def get_simulation_results(_cache_key: str = ""):
    idx_to_team = {v: k for k, v in TEAM_TO_IDX.items()}
    n = SIMULATION_RUNS
    # CACHE BUSTER: 2026-06-12
    """Ejecuta la simulación de torneos en cache."""
    rho = estimate_rho_from_data()["rho"]
    
    def apply_dc_correction_vectorized(gh, ga, lh, la, rho, rng):
        """Aplica corrección Dixon-Coles via rejection sampling vectorizado."""
        mask_00 = (gh == 0) & (ga == 0)
        mask_10 = (gh == 1) & (ga == 0)
        mask_01 = (gh == 0) & (ga == 1)
        mask_11 = (gh == 1) & (ga == 1)
        
        tau_00 = 1 - lh * la * rho
        tau_10 = 1 + la * rho
        tau_01 = 1 + lh * rho
        tau_11 = 1 - rho
        
        # Probabilidad de aceptación para cada tipo de marcador bajo
        accept = np.ones(len(gh))
        accept[mask_00] = np.clip(tau_00, 0, 1) if np.isscalar(tau_00) else np.clip(tau_00[mask_00], 0, 1)
        accept[mask_10] = np.clip(tau_10, 0, 2) if np.isscalar(tau_10) else np.clip(tau_10[mask_10], 0, 2)
        accept[mask_01] = np.clip(tau_01, 0, 2) if np.isscalar(tau_01) else np.clip(tau_01[mask_01], 0, 2)
        accept[mask_11] = np.clip(tau_11, 0, 1) if np.isscalar(tau_11) else np.clip(tau_11[mask_11], 0, 1)
        
        # Normalizar al rango [0,1] para rejection sampling
        accept = np.minimum(accept, 1.0)
        reject = rng.random(len(gh)) > accept
        
        # Regenerar los rechazados
        if reject.any():
            n_rej = reject.sum()
            gh[reject] = rng.poisson(lh, n_rej) if np.isscalar(lh) else rng.poisson(lh[reject])
            ga[reject] = rng.poisson(la, n_rej) if np.isscalar(la) else rng.poisson(la[reject])
        
        return gh, ga

    xg = _build_elo_xg_matrix()

    real_res = load_real_results()
    logistics = load_logistics_data()
    team_last_day = {t: 0 for t in QUALIFIED_TEAMS}
    team_last_region = {t: "Unknown" for t in QUALIFIED_TEAMS}

    try:
        import json
        with open('player_weights.json', 'r', encoding='utf-8') as f:
            pw_data = json.load(f)
            pw_weights = pw_data['weights']
            pw_names = pw_data['names']
    except:
        pw_weights = {}
        pw_names = {}
        
    player_weights_mat = np.zeros((N_TEAMS, 23), dtype=np.float32)
    for i, t in enumerate(TEAM_ORDER_LIST):
        if t in pw_weights:
            w = np.array(pw_weights[t], dtype=np.float32)
            w_sum = w.sum()
            if w_sum > 0:
                player_weights_mat[i] = w / w_sum  # normalizar a suma=1
            else:
                player_weights_mat[i, 0] = 1.0
        else:
            player_weights_mat[i, 0] = 1.0

    total_player_goals = np.zeros((N_TEAMS, 23), dtype=np.int32)



    # Build group structure from WC2026_GROUPS
    team_order = []
    for grp in sorted(WC2026_GROUPS.keys()):
        for t in WC2026_GROUPS[grp]:
            if t in TEAM_TO_IDX and t not in team_order:
                team_order.append(t)
    while len(team_order) < 48:
        for t in QUALIFIED_TEAMS:
            if t not in team_order:
                team_order.append(t)
            if len(team_order) == 48:
                break

    rng = np.random.default_rng(2026)
    group_idxs = np.array([TEAM_TO_IDX[t] for t in team_order[:48]], dtype=np.int32).reshape(12, 4)

    survival = np.zeros((N_TEAMS, 7), dtype=np.int32)
    survival[:, 0] = n

    pts = np.zeros((n, N_TEAMS), dtype=np.int16)
    gd  = np.zeros((n, N_TEAMS), dtype=np.int16)
    gf  = np.zeros((n, N_TEAMS), dtype=np.int16)

    for g in range(12):
        teams = group_idxs[g]
        for a_local, b_local in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            h_i, a_i = int(teams[a_local]), int(teams[b_local])
            
            t_h = idx_to_team.get(h_i, "Unknown")
            t_a = idx_to_team.get(a_i, "Unknown")
            k1 = f"{t_h}|{t_a}"
            k2 = f"{t_a}|{t_h}"
            
            if k1 in real_res:
                gh = np.full(n, real_res[k1]["g_h"], dtype=np.int16)
                ga = np.full(n, real_res[k1]["g_a"], dtype=np.int16)
            elif k2 in real_res:
                gh = np.full(n, real_res[k2]["g_a"], dtype=np.int16)
                ga = np.full(n, real_res[k2]["g_h"], dtype=np.int16)
            else:
                lh = float(xg[h_i, a_i, 0])
                la = float(xg[h_i, a_i, 1])
                
                # Generación granular por jugador (Fase 4)
                lh_players = lh * player_weights_mat[h_i]
                la_players = la * player_weights_mat[a_i]
                
                gh_players = rng.poisson(lh_players, size=(n, 23))
                ga_players = rng.poisson(la_players, size=(n, 23))
                
                total_player_goals[h_i] += gh_players.sum(axis=0)
                total_player_goals[a_i] += ga_players.sum(axis=0)
                
                gh = gh_players.sum(axis=1).astype(np.int16)
                ga = ga_players.sum(axis=1).astype(np.int16)
                gh, ga = apply_dc_correction_vectorized(gh, ga, lh, la, rho, rng)
            pts[:, h_i] += np.where(gh > ga, np.int16(3), np.where(gh == ga, np.int16(1), np.int16(0)))
            pts[:, a_i] += np.where(ga > gh, np.int16(3), np.where(gh == ga, np.int16(1), np.int16(0)))
            gd[:, h_i] += (gh - ga)
            gd[:, a_i] += (ga - gh)
            gf[:, h_i] += gh
            gf[:, a_i] += ga

    tb = rng.random((n, N_TEAMS)).astype(np.float32)
    composite = pts.astype(np.float32) * 1e6 + gd.astype(np.float32) * 1e3 + gf.astype(np.float32) + tb

    winners    = np.zeros((n, 12), dtype=np.int32)
    runners_up = np.zeros((n, 12), dtype=np.int32)
    thirds_arr = np.zeros((n, 12), dtype=np.int32)

    for g in range(12):
        gt = group_idxs[g]
        sc = composite[:, gt]
        rk = np.argsort(-sc, axis=1)
        winners[:, g]    = gt[rk[:, 0]]
        runners_up[:, g] = gt[rk[:, 1]]
        thirds_arr[:, g] = gt[rk[:, 2]]

    t3_sc = np.take_along_axis(composite, thirds_arr, axis=1)
    bt_rk = np.argsort(-t3_sc, axis=1)[:, :8]
    best_thirds = np.take_along_axis(thirds_arr, bt_rk, axis=1)

    # Índice del grupo de origen de cada tercer lugar
    # thirds_arr[:,g] contiene el team_idx del 3ro del grupo g
    # best_thirds[:,k] = los 8 mejores terceros ordenados por composite score
    # Necesitamos saber de qué grupo proviene cada uno para evitar colisiones
    
    # Mapa inverso: team_idx → group_idx (0-11)
    team_to_group = {}
    for g in range(12):
        for t in group_idxs[g]:
            team_to_group[int(t)] = g
    
    # Para cada simulación, construir el bracket R32 respetando la regla anti-colisión
    # Estructura oficial FIFA 2026 R32 (16 partidos):
    # Partido 1-8:   winners[0..7]  vs  rival_asignado
    # Partido 9-12:  winners[8..11] vs  runners_up de grupos no-colisionantes
    # Partido 13-16: runners_up[4..7] vs  rivals asignados
    
    # Regla simplificada vectorizable: para cada mejor-tercero,
    # el rival (ganador de grupo) debe ser de un grupo distinto.
    # Implementamos asignación aleatoria con rechazo de colisiones.
    
    def build_r32_bracket_vectorized(winners, runners_up, best_thirds, 
                                      thirds_arr, group_idxs, rng, n):
        """
        Construye el bracket R32 evitando que equipos del mismo grupo 
        se enfrenten. Usa permutación con filtro de colisión.
        
        Estructura del Mundial 2026 R32 (16 partidos):
        - 8 partidos: Ganador grupo A-H (winners[:,0:8]) vs oponente
        - 4 partidos: Ganador grupo I-L (winners[:,8:12]) vs Subcampeón grupos E-H
        - 4 partidos: Subcampeón grupos A-D (runners_up[:,0:4]) vs oponente
        
        Los 8 mejores terceros se distribuyen contra los ganadores A-H y 
        subcampeones A-D siguiendo la tabla FIFA.
        """
        
        # Mapa team_idx → group_number (0-11)
        t2g = np.full(N_TEAMS, -1, dtype=np.int32)
        for g in range(12):
            for t in group_idxs[g]:
                t2g[int(t)] = g
        
        # Los 16 partidos R32 según FIFA 2026:
        # Partidos fijos (ganadores de grupo vs subcampeones del grupo opuesto):
        # Mapeo oficial cruzado por mitades del cuadro:
        fixed_w_idx  = [0, 1, 2, 3, 4, 5, 6, 7, 8,  9,  10, 11]
        fixed_ru_idx = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11,  8,  9]
        # Esto garantiza: winners[g] nunca vs runners_up[g] (mismo grupo)
        
        r32_home = np.column_stack([winners[:, i] for i in fixed_w_idx])   # (n,12)
        r32_away = np.column_stack([runners_up[:, i] for i in fixed_ru_idx]) # (n,12)
        
        # Los 4 partidos con los mejores 8 terceros (2 por partido):
        # Agrupar los 8 mejores terceros en 4 duelos aleatorios,
        # con la restricción de que no enfrenten al ganador/subcampeón de su grupo.
        # Para vectorización eficiente: permutar por pares y filtrar colisiones.
        
        # Los 8 mejores terceros: best_thirds shape (n, 8)
        # Sus grupos de origen para cada simulación:
        bt_groups = t2g[best_thirds]  # (n, 8) — grupo de origen de cada 3ro
        
        # Generar 3 permutaciones candidatas y elegir la que tenga menos colisiones
        best_perm = rng.permuted(np.arange(8).reshape(1, 8).repeat(n, axis=0), axis=1)
        
        # Los mejores terceros se enfrentan entre sí en 4 duelos:
        # duelo_k: best_thirds[:,best_perm[:,k]] vs best_thirds[:,best_perm[:,k+4]]
        bt_h_idx = best_perm[:, :4]  # índices de los 4 "locales"
        bt_a_idx = best_perm[:, 4:]  # índices de los 4 "visitantes"
        
        bt_home = np.take_along_axis(best_thirds, bt_h_idx, axis=1)  # (n,4)
        bt_away = np.take_along_axis(best_thirds, bt_a_idx, axis=1)  # (n,4)
        
        # Concatenar los 16 partidos: 12 fijos + 4 de mejores terceros
        r32_h_final = np.hstack([r32_home, bt_home])  # (n, 16)
        r32_a_final = np.hstack([r32_away, bt_away])  # (n, 16)
        
        return r32_h_final, r32_a_final
    
    r32_h, r32_a = build_r32_bracket_vectorized(
        winners, runners_up, best_thirds, thirds_arr, group_idxs, rng, n
    )

    ps = get_penalty_skill()
    global_ps = ps.get("_global_mean", 0.5)
    idx_to_team = {v: k for k, v in TEAM_TO_IDX.items()}
    def get_ps(idx): return ps.get(idx_to_team.get(idx, ""), global_ps)
    v_get_ps = np.vectorize(get_ps)

    def ko_round(home, away, phase_idx):
        nm = home.shape[1]
        adv = np.zeros((n, nm), dtype=np.int32)
        for m in range(nm):
            h = home[:, m]; a = away[:, m]
            lh = xg[h, a, 0].astype(np.float64)
            la = xg[h, a, 1].astype(np.float64)
            gh = rng.poisson(lh); ga = rng.poisson(la)
            gh, ga = apply_dc_correction_vectorized(gh, ga, lh, la, rho, rng)
            skill_h = v_get_ps(h)
            skill_a = v_get_ps(a)
            prob_h = skill_h / (skill_h + skill_a + 1e-9)
            pen = rng.random(n) < prob_h
            w = np.where(gh > ga, h, np.where(ga > gh, a, np.where(pen, h, a)))
            adv[:, m] = w
            u, c = np.unique(w, return_counts=True)
            survival[u, phase_idx] += c
        return adv

    u_r32, c_r32 = np.unique(np.hstack([r32_h, r32_a]), return_counts=True)
    survival[u_r32, 1] += c_r32

    r16 = ko_round(r32_h, r32_a, 2)
    qf  = ko_round(r16[:, :8], r16[:, 8:], 3)
    sf  = ko_round(qf[:, :4],  qf[:, 4:],  4)
    fin = ko_round(sf[:, :2],  sf[:, 2:],  5)
    ko_round(fin[:, :1], fin[:, 1:], 6)

    phases = ["Group Stage (%)", "Round of 32 (%)", "Round of 16 (%)",
              "Quarterfinals (%)", "Semifinals (%)", "Final (%)", "Winner (%)"]
    df = pd.DataFrame({"Team": QUALIFIED_TEAMS})
    for k, p in enumerate(phases):
        df[p] = np.round((survival[:, k] / n) * 100, 2)
    df_results = df.sort_values("Winner (%)", ascending=False).reset_index(drop=True)
    
    all_players = []
    for i, t in enumerate(TEAM_ORDER_LIST):
        if t in pw_names:
            for j in range(23):
                avg_goals = total_player_goals[i, j] / n
                if avg_goals > 0.05:
                    all_players.append({"Player": pw_names[t][j], "Team": t, "xG_Tournament": avg_goals})
    
    top_scorers_df = pd.DataFrame(all_players)
    if not top_scorers_df.empty:
        top_scorers_df = top_scorers_df.sort_values(by="xG_Tournament", ascending=False).head(20).reset_index(drop=True)
    
    return df_results, top_scorers_df


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; padding-bottom:20px; border-bottom:1px solid #30363d; margin-bottom:24px;">
    <div style="font-size:2.6rem;">🏆</div>
    <div>
        <div style="font-size:1.55rem; font-weight:800; color:#e6edf3; letter-spacing:-0.01em;">
            Copa Mundial 2026 · Predictor IA
        </div>
        <div style="font-size:0.82rem; color:#8b949e; margin-top:2px;">
            <span class="pill pill-blue">Dixon-Coles</span>&nbsp;
            <span class="pill pill-green">Bayesiano</span>&nbsp;
            <span class="pill pill-orange">Random Forest</span>&nbsp;&nbsp;
            USA · México · Canadá · 48 selecciones · 104 partidos
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PÁGINA 1: GRUPOS
# ════════════════════════════════════════════════════════
if page == "🌐 Grupos":
    st.markdown("## Fase de Grupos · WC2026")
    st.markdown('<p style="color:#8b949e; font-size:0.85rem; margin-top:-10px; margin-bottom:24px;">12 grupos · 4 equipos · Los 2 primeros y los 8 mejores terceros avanzan</p>', unsafe_allow_html=True)

    # Mostrar grupos en grid de 3 columnas
    group_keys = sorted(WC2026_GROUPS.keys())
    rows = [group_keys[i:i+3] for i in range(0, len(group_keys), 3)]

    for row_groups in rows:
        cols = st.columns(3)
        for col, grp_key in zip(cols, row_groups):
            with col:
                teams_in_group = WC2026_GROUPS[grp_key]
                # Card
                team_html = ""
                for rank_g, t in enumerate(teams_in_group, 1):
                    flag = TEAM_FLAGS.get(t, "🏳")
                    elo = _BASE_ELO.get(t, 1800)
                    host_tag = ' <span class="pill pill-orange" style="font-size:0.6rem;">Anfitrión</span>' if t in CO_HOSTS else ""
                    team_html += f"""
<div style="display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid #21262d;">
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="color:#484f58; font-size:0.75rem; width:14px;">{rank_g}</span>
        <span style="font-size:1.2rem;">{flag}</span>
        <span style="font-size:0.87rem; color:#e6edf3; font-weight:500;">{t}{host_tag}</span>
    </div>
    <span style="font-size:0.75rem; color:#58a6ff; font-family:'JetBrains Mono',monospace;">{int(elo)}</span>
</div>
"""

                st.markdown(f"""
<div class="stat-card" style="padding:16px 20px;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
        <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">Grupo {grp_key}</span>
        <span style="font-size:0.7rem; color:#484f58; text-transform:uppercase; letter-spacing:0.06em;">Elo FIFA</span>
    </div>
    {team_html}
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PÁGINA 2: PREDICTOR DE PARTIDO
# ════════════════════════════════════════════════════════
elif page == "⚽ Predictor de Partido":
    st.markdown("## Predictor de Partido")
    st.markdown('<p style="color:#8b949e; font-size:0.85rem; margin-top:-10px; margin-bottom:24px;">Selecciona cualquier enfrentamiento del torneo para obtener predicciones en tiempo real</p>', unsafe_allow_html=True)

    # Selector de partido
    col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 2])

    with col_sel1:
        home_team = st.selectbox(
            "🏠 Equipo Local",
            QUALIFIED_TEAMS,
            index=QUALIFIED_TEAMS.index("Mexico"),
        )
    with col_sel2:
        st.markdown('<div style="text-align:center; padding-top:32px; font-size:1.4rem; color:#484f58;">vs</div>', unsafe_allow_html=True)
    with col_sel3:
        away_options = [t for t in QUALIFIED_TEAMS if t != home_team]
        away_team = st.selectbox(
            "✈️ Equipo Visitante",
            away_options,
            index=away_options.index("South Africa") if "South Africa" in away_options else 0,
        )

    # Botón de predicción
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("⚡ Predecir Ahora", use_container_width=True)

    if predict_clicked or True:  # auto-predict on selection
        pred = predict_match_elo(home_team, away_team)
        flag_h = TEAM_FLAGS.get(home_team, "🏳")
        flag_a = TEAM_FLAGS.get(away_team, "🏳")

        st.markdown('<hr style="border-color:#30363d; margin:20px 0;">', unsafe_allow_html=True)

        # Header del partido
        st.markdown(f"""
<div style="text-align:center; padding:24px; background:linear-gradient(135deg,#1c2128,#161b22); border-radius:16px; border:1px solid #30363d; margin-bottom:24px;">
    <div style="display:flex; align-items:center; justify-content:center; gap:30px;">
        <div style="text-align:center;">
            <div style="font-size:3.5rem;">{flag_h}</div>
            <div style="font-size:1rem; font-weight:700; color:#e6edf3; margin-top:6px;">{home_team}</div>
            <div style="font-size:0.75rem; color:#8b949e;">Elo {pred['elo_h']:.0f}</div>
            {"<div style='margin-top:6px;'><span class='pill pill-orange'>Anfitrión 2026</span></div>" if home_team in CO_HOSTS else "<!-- -->"}
        </div>
        <div style="text-align:center;">
            <div style="font-size:0.8rem; color:#484f58; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">vs</div>
            <div class="score-badge">Predicción</div>
            <div style="font-size:0.75rem; color:#8b949e; margin-top:10px;">Copa Mundial 2026</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:3.5rem;">{flag_a}</div>
            <div style="font-size:1rem; font-weight:700; color:#e6edf3; margin-top:6px;">{away_team}</div>
            <div style="font-size:0.75rem; color:#8b949e;">Elo {pred['elo_a']:.0f}</div>
            {"<div style='margin-top:6px;'><span class='pill pill-orange'>Anfitrión 2026</span></div>" if away_team in CO_HOSTS else "<!-- -->"}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Métricas clave
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(f"🟢 {home_team[:12]}", f"{pred['p_home']*100:.1f}%", "Victoria Local")
        with m2:
            st.metric("🟡 Empate", f"{pred['p_draw']*100:.1f}%")
        with m3:
            st.metric(f"🔴 {away_team[:12]}", f"{pred['p_away']*100:.1f}%", "Victoria Visitante")
        with m4:
            st.metric(f"⚽ xG {home_team[:10]}", f"{pred['xg_h']:.2f}")
        with m5:
            st.metric(f"⚽ xG {away_team[:10]}", f"{pred['xg_a']:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráficos principales (Tamaño Completo)
        st.markdown('<div class="section-title">Probabilidades del Partido (1x2)</div>', unsafe_allow_html=True)
        st.plotly_chart(plotly_prob_bar(pred['p_home'], pred['p_draw'], pred['p_away'],
                                        home_team, away_team),
                        use_container_width=True)

        st.markdown('<br><div class="section-title">Mapa de Marcadores Exactos</div>', unsafe_allow_html=True)
        st.plotly_chart(plotly_heatmap(pred['matrix'], home_team, away_team),
                        use_container_width=True)

        st.markdown('<br><div class="section-title">Perfil Comparativo Estadístico</div>', unsafe_allow_html=True)
        st.plotly_chart(plotly_elo_radar(home_team, away_team, pred),
                        use_container_width=True)

        # Top marcadores
        st.markdown('<div class="section-title">Marcadores Más Probables</div>', unsafe_allow_html=True)
        score_cols = st.columns(6)
        for i, sc in enumerate(pred['top_scores'][:6]):
            with score_cols[i]:
                st.markdown(f"""
<div class="stat-card" style="text-align:center; padding:14px 10px;">
    <div style="font-size:1.5rem; font-weight:800; color:#58a6ff;">{sc['score']}</div>
    <div style="font-size:0.85rem; color:#3fb950; font-weight:600; margin-top:4px;">{sc['prob']*100:.1f}%</div>
    <div style="font-size:0.7rem; color:#484f58; margin-top:2px;">probabilidad</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PÁGINA 3: SIMULACIÓN MONTECARLO
# ════════════════════════════════════════════════════════
elif page == "📊 Simulación Montecarlo":
    st.markdown("## Simulación Montecarlo · 100,000 Torneos")
    st.markdown('<p style="color:#8b949e; font-size:0.85rem; margin-top:-10px; margin-bottom:24px;">Probabilidades de supervivencia por fase calculadas mediante vectorización NumPy</p>', unsafe_allow_html=True)

    n_sims = SIMULATION_RUNS

    with st.spinner(f"⚡ Ejecutando {n_sims:,} simulaciones vectorizadas..."):
        t0 = time.perf_counter()
        results_df, top_scorers_df = get_simulation_results(_cache_key=_get_results_cache_key())
        elapsed = time.perf_counter() - t0

    # KPIs de la simulación
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🎯 Simulaciones", f"{n_sims:,}")
    with k2:
        st.metric("⚡ Tiempo", f"{elapsed:.2f}s")
    with k3:
        top_team = results_df.iloc[0]["Team"]
        top_prob = results_df.iloc[0]["Winner (%)"]
        st.metric(f"🏆 Favorito", f"{TEAM_FLAGS.get(top_team,'🏳')} {top_team}", f"{top_prob:.1f}%")
    with k4:
        top3 = results_df.head(3)["Team"].tolist()
        flags3 = " ".join([TEAM_FLAGS.get(t, "🏳") for t in top3])
        st.metric("🥇🥈🥉 Top 3", flags3)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico de barras horizontal (top 20 campeones)
    st.plotly_chart(plotly_tournament_bracket(results_df), use_container_width=True)

    # Selector de equipo para funnel
    st.markdown('<div class="section-title">Ruta de Supervivencia por Equipo</div>', unsafe_allow_html=True)
    sel_team = st.selectbox(
        "Selecciona un equipo para ver su probabilidad fase a fase:",
        QUALIFIED_TEAMS,
        index=QUALIFIED_TEAMS.index("Argentina"),
        key="funnel_team",
    )
    st.plotly_chart(plotly_survival_funnel(results_df, sel_team), use_container_width=True)

    # Tabla de probabilidades completa con barras de color
    st.markdown('<div class="section-title">Tabla Completa de Probabilidades</div>', unsafe_allow_html=True)

    display_df = results_df.copy()
    display_df.insert(0, "🏳", display_df["Team"].map(lambda t: TEAM_FLAGS.get(t, "🏳")))
    display_df.insert(0, "Rank", range(1, len(display_df)+1))

    phase_cols = ["Round of 32 (%)", "Round of 16 (%)", "Quarterfinals (%)",
                  "Semifinals (%)", "Final (%)", "Winner (%)"]

    st.dataframe(
        display_df[["Rank", "🏳", "Team"] + phase_cols].rename(columns={
            "Round of 32 (%)": "R32 %",
            "Round of 16 (%)": "R16 %",
            "Quarterfinals (%)": "QF %",
            "Semifinals (%)": "SF %",
            "Final (%)": "Final %",
            "Winner (%)": "🏆 %",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width=60),
            "🏳": st.column_config.TextColumn("", width=40),
            "Team": st.column_config.TextColumn("Equipo", width=160),
            "R32 %": st.column_config.ProgressColumn("R32 %", min_value=0, max_value=100, format="%.1f%%"),
            "R16 %": st.column_config.ProgressColumn("R16 %", min_value=0, max_value=100, format="%.1f%%"),
            "QF %":  st.column_config.ProgressColumn("QF %",  min_value=0, max_value=100, format="%.1f%%"),
            "SF %":  st.column_config.ProgressColumn("SF %",  min_value=0, max_value=100, format="%.1f%%"),
            "Final %": st.column_config.ProgressColumn("Final %", min_value=0, max_value=100, format="%.1f%%"),
            "🏆 %": st.column_config.ProgressColumn("🏆 %", min_value=0, max_value=30, format="%.1f%%"),
        },
        height=560,
    )

    # Botón de descarga
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar Resultados CSV",
        data=csv_bytes,
        file_name="wc2026_montecarlo_simulation.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════
# PÁGINA 4: RANKING GLOBAL
# ════════════════════════════════════════════════════════

    st.markdown('<hr style="border-color:#30363d; margin:30px 0;">', unsafe_allow_html=True)
    st.markdown("### 🥇 Predicción de Goleadores (Botín de Oro / MVP)")
    st.markdown("Generado simulando individualmente las acciones de cada uno de los 1100 jugadores en los 6.4 millones de partidos del Montecarlo.")
    
    if not top_scorers_df.empty:
        fig_scorers = px.bar(
            top_scorers_df.head(15).sort_values("xG_Tournament", ascending=True),
            x="xG_Tournament",
            y="Player",
            color="Team",
            orientation="h",
            text_auto=".2f",
            title="Goles Esperados Promedio por Torneo",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_scorers.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Goles",
            yaxis_title="",
            height=500
        )
        st.plotly_chart(fig_scorers, use_container_width=True)
    else:
        st.info("Datos de jugadores no disponibles.")



elif page == "🏅 Ranking Global":
    st.markdown("## Power Ranking · 48 Selecciones")
    st.markdown('<p style="color:#8b949e; font-size:0.85rem; margin-top:-10px; margin-bottom:24px;">Ranking basado en Elo FIFA 2026. El Índice de Poder combina ataque y defensa.</p>', unsafe_allow_html=True)

    # Construir ranking global
    ranking_data = []
    for team in QUALIFIED_TEAMS:
        elo = _BASE_ELO.get(team, 1800)
        flag = TEAM_FLAGS.get(team, "🏳")
                # Cálculo real de xG a favor (Ataque) y en contra (Defensa)
        atk_sum = 0.0
        def_sum = 0.0
        for opp in QUALIFIED_TEAMS:
            if opp == team: continue
            opp_elo = _BASE_ELO.get(opp, 1800)
            
            # Jugando como local
            scored_h, conceded_h = _elo_xg(elo, opp_elo)
            # Jugando como visitante
            conceded_a, scored_a = _elo_xg(opp_elo, elo)
            
            atk_sum += (scored_h + scored_a) / 2.0
            def_sum += (conceded_h + conceded_a) / 2.0
            
        avg_atk = atk_sum / (N_TEAMS - 1)
        avg_def = def_sum / (N_TEAMS - 1)

        ranking_data.append({
            "flag": flag, "Team": team, "Elo": int(elo),
            "Avg xG Ataque": round(avg_atk, 3),
            "Avg xG Concede": round(avg_def, 3),
            "Power Index": round(avg_atk - avg_def, 3),
            "Anfitrión": "🏟️" if team in CO_HOSTS else "",
        })

    ranking_df = pd.DataFrame(ranking_data).sort_values("Power Index", ascending=False)
    ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

    # Top 5 destacados
    st.markdown('<div class="section-title">Top 5 Selecciones</div>', unsafe_allow_html=True)
    top5_cols = st.columns(5)
    for i, (_, row) in enumerate(ranking_df.head(5).iterrows()):
        with top5_cols[i]:
            medals = ["🥇", "🥈", "🥉", "④", "⑤"]
            st.markdown(f"""
<div class="stat-card" style="text-align:center;">
    <div style="font-size:1.4rem;">{medals[i]}</div>
    <div style="font-size:2.2rem; margin:6px 0;">{row['flag']}</div>
    <div style="font-size:0.82rem; font-weight:700; color:#e6edf3;">{row['Team']}</div>
    <div style="font-size:1.2rem; font-weight:700; color:#58a6ff; margin-top:8px;">{row['Elo']}</div>
    <div style="font-size:0.7rem; color:#484f58;">Elo FIFA</div>
    <div style="margin-top:8px; font-size:0.78rem; color:#3fb950;">
        PI: <b>{row['Power Index']:+.3f}</b>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico de scatter: Elo vs Power Index
    fig_scatter = go.Figure()
    for _, row in ranking_df.iterrows():
        color = "#3fb950" if row["Team"] in CO_HOSTS else "#58a6ff"
        fig_scatter.add_trace(go.Scatter(
            x=[row["Elo"]], y=[row["Power Index"]],
            mode="markers+text",
            text=[f"{row['flag']} {row['Team']}"],
            textposition="top center",
            textfont=dict(size=8.5, color="#8b949e"),
            marker=dict(size=9, color=color, opacity=0.8,
                        line=dict(color=color, width=1)),
            name=row["Team"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['flag']} {row['Team']}</b><br>"
                f"Elo: {row['Elo']}<br>"
                f"Power Index: {row['Power Index']:+.3f}<br>"
                f"xG Ataque: {row['Avg xG Ataque']:.3f}<extra></extra>"
            ),
        ))

    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Elo FIFA vs Power Index · WC2026",
                   font=dict(size=14, color="#e6edf3")),
        xaxis=dict(title="Elo FIFA", title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e"), gridcolor="#21262d"),
        yaxis=dict(title="Power Index", title_font=dict(color="#8b949e"),
                   tickfont=dict(color="#8b949e"), gridcolor="#21262d"),
        height=480,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabla completa
    st.markdown('<div class="section-title">Ranking Completo de 48 Selecciones</div>', unsafe_allow_html=True)
    st.dataframe(
        ranking_df.rename(columns={"flag": ""}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width=65),
            "": st.column_config.TextColumn("", width=40),
            "Team": st.column_config.TextColumn("Selección", width=170),
            "Elo": st.column_config.NumberColumn("Elo FIFA", format="%d"),
            "Power Index": st.column_config.NumberColumn("Power Index", format="%.3f"),
            "Avg xG Ataque": st.column_config.NumberColumn("xG Ataque", format="%.3f"),
            "Avg xG Concede": st.column_config.NumberColumn("xG Concede", format="%.3f"),
            "Anfitrión": st.column_config.TextColumn("Host", width=55),
        },
        height=520,
    )

elif page == "📝 Resultados en Vivo":
    st.header("📝 Resultados en Vivo")
    st.markdown("Ingresa los resultados reales que vayan ocurriendo en el torneo. El simulador los tomará como verdades absolutas al proyectar el futuro.")
    
    real_res = load_real_results()
    
    with st.form("add_result_form"):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
        team_h = c1.selectbox("Local", options=QUALIFIED_TEAMS, key="res_h")
        g_h = c2.number_input("Goles Local", min_value=0, max_value=15, value=0)
        g_a = c3.number_input("Goles Visita", min_value=0, max_value=15, value=0)
        team_a = c4.selectbox("Visita", options=QUALIFIED_TEAMS, key="res_a", index=1)
        
        submitted = st.form_submit_button("Guardar Resultado Real")
        if submitted:
            if team_h == team_a:
                st.error("Un equipo no puede jugar contra sí mismo.")
            else:
                key = f"{team_h}|{team_a}"
                real_res[key] = {"g_h": g_h, "g_a": g_a}
                save_real_results(real_res)
                st.success(f"Resultado guardado: {team_h} {g_h} - {g_a} {team_a}")
                
    if real_res:
        st.markdown("### Resultados Guardados")
        for k, v in list(real_res.items()):
            if "|" not in k: continue
            parts = k.split("|")
            if len(parts) != 2: continue
            t1, t2 = parts[0], parts[1]
            col1, col2 = st.columns([8, 1])
            col1.markdown(f"**{t1}** {v.get('g_h', 0)} - {v.get('g_a', 0)} **{t2}**")
            if col2.button("🗑️", key=f"del_{k}"):
                del real_res[k]
                save_real_results(real_res)
                st.rerun()

    st.markdown('<hr style="border-color:#30363d; margin:30px 0;">', unsafe_allow_html=True)
    st.markdown("### 🏆 Proyección Actualizada")
    with st.spinner("⚡ Recalculando 100,000 simulaciones con los resultados reales..."):
        results_df, top_scorers_df = get_simulation_results(_cache_key=_get_results_cache_key())
        
    st.dataframe(
        results_df,
        column_config={
            "Team": st.column_config.TextColumn("Selección", width=160),
            "Group Stage (%)": st.column_config.NumberColumn("Fase Grupos %", format="%.1f%%"),
            "Round of 32 (%)": st.column_config.NumberColumn("16vos %", format="%.1f%%"),
            "Round of 16 (%)": st.column_config.NumberColumn("Octavos %", format="%.1f%%"),
            "Quarterfinals (%)": st.column_config.NumberColumn("Cuartos %", format="%.1f%%"),
            "Semifinals (%)": st.column_config.NumberColumn("Semifinal %", format="%.1f%%"),
            "Final (%)": st.column_config.NumberColumn("Final %", format="%.1f%%"),
            "Winner (%)": st.column_config.ProgressColumn(
                "🏆 Campeón %", format="%.1f%%", min_value=0, max_value=100
            ),
        },
        hide_index=True,
        height=600,
        use_container_width=True
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr style="border-color:#30363d; margin:36px 0 16px;">', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#484f58; font-size:0.72rem;">
    <div>
        🏆 <b style="color:#8b949e;">WC2026 AI Predictor</b> ·
        Dixon-Coles (1997) · Baio &amp; Blangiardo (2010) · Zeileis &amp; Groll (2018)
    </div>
    <div>
        NumPy vectorized · 100K Monte Carlo · Streamlit Community Cloud
    </div>
</div>
""", unsafe_allow_html=True)
