import os
import pandas as pd
import time

QUALIFIED_TEAMS = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR", "Czechia": "CZE", "Canada": "CAN", 
    "Switzerland": "SUI", "Qatar": "QAT", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Morocco": "MAR", 
    "Haiti": "HAI", "Scotland": "SCO", "United States": "USA", "Paraguay": "PAR", "Australia": "AUS", 
    "Türkiye": "TUR", "Germany": "GER", "Curaçao": "CUW", "Ivory Coast": "CIV", "Ecuador": "ECU", 
    "Netherlands": "NED", "Japan": "JPN", "Tunisia": "TUN", "Sweden": "SWE", "Belgium": "BEL", 
    "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL", "Spain": "ESP", "Cabo Verde": "CPV", 
    "Saudi Arabia": "KSA", "Uruguay": "URU", "France": "FRA", "Senegal": "SEN", "Norway": "NOR", 
    "Iraq": "IRQ", "Argentina": "ARG", "Algeria": "ALG", "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "POR", "Uzbekistan": "UZB", "Colombia": "COL", "Congo DR": "COD", "England": "ENG", 
    "Croatia": "CRO", "Ghana": "GHA", "Panama": "PAN"
}

print("=== INICIANDO MOTOR TÁCTICO DUAL (KAGGLE: 80% ACTUAL / 20% HISTÓRICO) ===")

# 1. Cargar el dataset de Kaggle
try:
    df = pd.read_csv('Top5_League_Players_2017to2024_dataset.csv', sep=';', on_bad_lines='skip', encoding='utf-8')
except Exception as e:
    print(f"Error cargando CSV: {e}")
    exit(1)

# Limpiar columnas
df['Playing Time_Min'] = pd.to_numeric(df['Playing Time_Min'], errors='coerce').fillna(0)
df['Performance_Gls'] = pd.to_numeric(df['Performance_Gls'], errors='coerce').fillna(0)
df['Performance_Ast'] = pd.to_numeric(df['Performance_Ast'], errors='coerce').fillna(0)
df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0)

# 2. Separar Actual (2324, 2425) vs Histórico (< 2324)
df_current = df[df['season'] >= 2324].copy()
df_history = df[df['season'] < 2324].copy()

sqi_data = []

for team_name, nation_code in QUALIFIED_TEAMS.items():
    # Filtrar jugadores de esta nación
    nat_curr = df_current[df_current['nation_'] == nation_code]
    nat_hist = df_history[df_history['nation_'] == nation_code]
    
    current_pts = 0
    history_pts = 0
    
    # 80% ACTUAL: Evaluar a los 20 mejores jugadores de la temporada actual
    if not nat_curr.empty:
        # Agrupar por jugador (por si jugó en 2 equipos en la misma temporada)
        top_curr = nat_curr.groupby('player').sum(numeric_only=True).sort_values('Playing Time_Min', ascending=False).head(20)
        for _, row in top_curr.iterrows():
            mins = row['Playing Time_Min']
            gls = row['Performance_Gls']
            ast = row['Performance_Ast']
            
            pts = (mins / 100.0) + (gls * 3.0) + (ast * 2.0)
            current_pts += pts

    # 20% HISTÓRICO: Evaluar a los 20 mejores jugadores históricamente
    if not nat_hist.empty:
        top_hist = nat_hist.groupby('player').sum(numeric_only=True).sort_values('Playing Time_Min', ascending=False).head(20)
        for _, row in top_hist.iterrows():
            mins = row['Playing Time_Min']
            gls = row['Performance_Gls']
            ast = row['Performance_Ast']
            
            pts = (mins / 100.0) + (gls * 3.0) + (ast * 2.0)
            history_pts += pts

    # Normalizar (Los equipos élite sacarán ~1000 puntos actuales y ~3000 puntos históricos)
    curr_score = min(1.0, current_pts / 800.0)
    hist_score = min(1.0, history_pts / 2400.0)
    
    # Si el equipo no tiene jugadores en el Top 5 Leagues, asume base 0.85 (para no destruir su Elo base)
    if current_pts == 0 and history_pts == 0:
        final_sqi = 1.0  # Neutral, se queda solo con el Elo
    else:
        # Ponderación 80/20 del usuario
        tactical_power = (curr_score * 0.80) + (hist_score * 0.20)
        
        # Escalar a multiplicador Elo (0.85 a 1.25)
        final_sqi = 0.85 + (tactical_power * 0.40)
        final_sqi = min(1.25, max(0.85, final_sqi))
        
    sqi_data.append({
        "Team": team_name, 
        "SQI": round(final_sqi, 3), 
        "Pts_Actual": round(current_pts, 1),
        "Pts_Hist": round(history_pts, 1)
    })
    
    print(f"[{team_name}] SQI: {final_sqi:.3f} | Actual: {current_pts:.0f} pts | Hist: {history_pts:.0f} pts")

df_sqi = pd.DataFrame(sqi_data)
df_sqi.to_csv("fbref_sqi.csv", index=False)
print("\n[+] Base de datos guardada en 'fbref_sqi.csv'")
