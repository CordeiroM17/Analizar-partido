"""
validation.py (video)
=====================
Validación del tracking contra la verdad oficial de SofaScore.

SofaScore actúa como set de validación de la extracción del video:
  - sofascore/lineups_clean.csv    -> cuántos jugadores jugaron y cuántos minutos
  - sofascore/statistics_clean.csv -> totales que el tracking debería aproximar

Si los números del tracking se alejan mucho de SofaScore, la extracción
(homografía, tracking o clasificación de equipos) tiene problemas.
"""

import pandas as pd


def track_summary(df_positions, sample_fps=5):
    """
    Resumen por track_id: equipo, frames detectados, minutos estimados
    y posición promedio (para inspeccionar identidades).
    """
    if df_positions is None or df_positions.empty:
        return pd.DataFrame()

    df = df_positions[df_positions["role"] != "referee"].copy()
    g = (df.groupby(["track_id", "team"])
         .agg(frames=("frame", "nunique"),
              first_sec=("t_sec", "min"),
              last_sec=("t_sec", "max"),
              avg_x=("x", "mean"),
              avg_y=("y", "mean"),
              avg_conf=("confidence", "mean"))
         .reset_index())
    g["est_minutes"] = ((g["last_sec"] - g["first_sec"]) / 60).round(1)
    g["avg_x"] = g["avg_x"].round(1)
    g["avg_y"] = g["avg_y"].round(1)
    g["avg_conf"] = g["avg_conf"].round(2)
    return g.sort_values("frames", ascending=False).reset_index(drop=True)


def players_per_frame(df_positions):
    """
    Jugadores detectados por frame y por equipo. Lo esperable es ~11 por
    equipo (10-11 con cámara de TV que no toma toda la cancha; menos si hubo
    una expulsión — ver incidents_clean.csv antes de alarmarse).
    Retorna DataFrame: team, mean, p10, p90.
    """
    if df_positions is None or df_positions.empty:
        return pd.DataFrame()

    df = df_positions[df_positions["role"] != "referee"]
    counts = df.groupby(["frame", "team"]).size().reset_index(name="n")
    return (counts.groupby("team")["n"]
            .agg(mean="mean",
                 p10=lambda s: s.quantile(0.10),
                 p90=lambda s: s.quantile(0.90))
            .round(1)
            .reset_index())


def compare_with_lineups(df_positions, df_lineups, is_home=True, sample_fps=5):
    """
    Cruza el tracking con lineups de SofaScore (sofascore/lineups_clean.csv).

    Comparaciones (por equipo del video vs lineups):
      - jugadores distintos detectados vs jugadores que realmente jugaron
      - minutos-jugador totales estimados vs minutos oficiales

    NOTA: los track_ids se fragmentan (un jugador puede tener varios tracks),
    así que 'tracks' >= 'jugadores reales' es normal. Mucho más del doble
    indica tracking inestable.

    Retorna dict con los números de ambas fuentes y banderas de alerta.
    """
    if df_positions is None or df_positions.empty:
        return None
    if df_lineups is None or df_lineups.empty:
        return None

    lin = df_lineups.copy()
    lin["minutes_played"] = pd.to_numeric(lin["minutes_played"], errors="coerce").fillna(0)
    played = lin[lin["minutes_played"] > 0]

    official = {
        side: {
            "jugadores": int((played["team_side"] == side).sum()),
            "minutos_totales": int(played.loc[played["team_side"] == side,
                                              "minutes_played"].sum()),
        }
        for side in ("home", "away")
    }

    ts = track_summary(df_positions, sample_fps)
    tracking = {}
    for team_id in sorted(ts["team"].unique()):
        if team_id == -1:
            continue
        sub = ts[ts["team"] == team_id]
        tracking[int(team_id)] = {
            "tracks": len(sub),
            "minutos_totales_est": float(sub["est_minutes"].sum().round(1)),
        }

    alerts = []
    for side, off in official.items():
        # Sin mapeo camiseta->equipo aún, comparamos contra ambos clusters
        for team_id, trk in tracking.items():
            ratio = trk["tracks"] / max(off["jugadores"], 1)
            if ratio > 2.5:
                alerts.append(
                    f"Cluster {team_id}: {trk['tracks']} tracks vs "
                    f"{off['jugadores']} jugadores ({side}) — tracking muy fragmentado")

    expected_total = sum(o["minutos_totales"] for o in official.values())
    tracked_total = sum(t["minutos_totales_est"] for t in tracking.values())
    if tracked_total < expected_total * 0.5:
        alerts.append(
            f"Minutos trackeados ({tracked_total:.0f}) muy por debajo de los "
            f"oficiales ({expected_total}) — cobertura del video incompleta")

    return {
        "oficial_sofascore": official,
        "tracking_video": tracking,
        "alertas": alerts,
    }


def positions_plausibility(df_positions):
    """
    Chequeos de plausibilidad espacial de la homografía:
      - % de puntos dentro de la cancha (0-100)
      - dispersión: si todos los puntos caen en una franja chica,
        la homografía probablemente está mal.
    """
    if df_positions is None or df_positions.empty:
        return None

    x = pd.to_numeric(df_positions["x"], errors="coerce")
    y = pd.to_numeric(df_positions["y"], errors="coerce")
    inside = ((x >= 0) & (x <= 100) & (y >= 0) & (y <= 100)).mean()

    return {
        "pct_dentro_cancha": round(float(inside) * 100, 1),
        "x_rango": (round(float(x.min()), 1), round(float(x.max()), 1)),
        "y_rango": (round(float(y.min()), 1), round(float(y.max()), 1)),
        "x_std": round(float(x.std()), 1),
        "y_std": round(float(y.std()), 1),
        "ok": bool(inside > 0.95 and x.std() > 15 and y.std() > 10),
    }
