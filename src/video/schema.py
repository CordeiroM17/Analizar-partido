"""
schema.py (video)
=================
Contratos de datos del pipeline de video.

Decisión de diseño clave: TODAS las coordenadas de salida usan el sistema
OPTA 0-100 (igual que SofaScore). Así, cualquier dato derivado del video se
grafica y se cruza directamente con la data de SofaScore sin conversiones.

Archivos que produce el pipeline en data/.../{match_key}/tracking/:

  positions.csv      — una fila por (frame muestreado, jugador detectado)
  ball_positions.csv — una fila por frame muestreado donde se detectó la pelota
  tracking_meta.json — metadata de la corrida (video, fps, modelo, fechas)

Ver docs/DISENO_eventos.md para el esquema de events/events.csv (capas L4-L6).
"""

import os
import json
import pandas as pd

import config


# Columnas obligatorias de positions.csv
POSITIONS_COLUMNS = [
    "frame",       # nro de frame original del video
    "t_sec",       # segundo del video (frame / fps)
    "track_id",    # id persistente del jugador (ByteTrack)
    "role",        # 'player' | 'goalkeeper' | 'referee'
    "team",        # 0 | 1 (cluster de camiseta) | -1 (árbitro/desconocido)
    "x",           # coordenada OPTA 0-100 (largo de la cancha)
    "y",           # coordenada OPTA 0-100 (ancho de la cancha)
    "confidence",  # confianza de la detección
]

BALL_COLUMNS = ["frame", "t_sec", "x", "y", "confidence"]


def empty_positions_df():
    return pd.DataFrame(columns=POSITIONS_COLUMNS)


def validate_positions_df(df):
    """
    Valida el contrato de positions.csv.
    Retorna lista de problemas (vacía = todo OK). Distingue problemas FATALES
    (dataframe vacío, faltan columnas) de problemas de CALIDAD (valores fuera
    de rango) -- estos últimos son esperables en algunos frames (homografía
    fallida en repeticiones, primeros planos, cancha con líneas borrosas) y
    no deberían tirar todo el resultado a la basura; ver save_tracking().
    """
    problems = []
    if df is None or df.empty:
        return ["FATAL: DataFrame vacío"]

    missing = [c for c in POSITIONS_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"FATAL: Faltan columnas: {missing}")
        return problems

    for col in ("x", "y"):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            problems.append(f"'{col}' tiene valores no numéricos")
        out = vals.dropna()
        if not out.empty and ((out < -5) | (out > 105)).any():
            n_bad = int(((out < -5) | (out > 105)).sum())
            pct = 100 * n_bad / len(out)
            problems.append(
                f"'{col}' tiene {n_bad}/{len(out)} valores fuera de rango "
                f"OPTA ({pct:.1f}%) -- probable homografía fallida en esos frames")

    if pd.to_numeric(df["t_sec"], errors="coerce").isna().any():
        problems.append("'t_sec' tiene valores no numéricos")

    return problems


def save_tracking(df_positions, df_ball, meta, match_key, season=None):
    """
    Guarda los outputs del pipeline en la carpeta tracking/ del partido.

    Valida el contrato, pero solo aborta el guardado ante problemas FATALES
    (vacío / faltan columnas). Problemas de calidad (posiciones fuera de
    rango en algunos frames -- esperable, homografía no es perfecta en el
    100% de los frames) se listan como advertencia y quedan en el CSV: es
    tracking crudo, la limpieza/filtrado es tarea de una etapa posterior
    (F1 motor de posesión / revisión), no del guardado. Nunca se descarta
    una corrida completa -- puede representar horas de cómputo -- por esto.
    """
    problems = validate_positions_df(df_positions)
    fatal = [p for p in problems if p.startswith("FATAL")]
    if fatal:
        raise ValueError(f"positions no cumple el contrato: {fatal}")
    for p in problems:
        print(f"⚠️  {p}")

    out_dir = config.match_tracking_dir(match_key, season)
    os.makedirs(out_dir, exist_ok=True)

    df_positions.to_csv(os.path.join(out_dir, "positions.csv"), index=False)
    if df_ball is not None and not df_ball.empty:
        df_ball.to_csv(os.path.join(out_dir, "ball_positions.csv"), index=False)
    with open(os.path.join(out_dir, "tracking_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Guardado final OK -- los checkpoints de emergencia ya no hacen falta.
    for ckpt in ("positions.checkpoint.csv", "ball_positions.checkpoint.csv"):
        ckpt_path = os.path.join(out_dir, ckpt)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    return out_dir


def load_tracking(match_key, season=None):
    """
    Carga los datos de tracking de un partido.
    Retorna dict {'positions', 'ball', 'meta'} (None donde no haya archivo).
    """
    tdir = config.match_tracking_dir(match_key, season)

    def _csv(name):
        p = os.path.join(tdir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    meta = None
    mpath = os.path.join(tdir, "tracking_meta.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "positions": _csv("positions.csv"),
        "ball":      _csv("ball_positions.csv"),
        "meta":      meta,
    }
