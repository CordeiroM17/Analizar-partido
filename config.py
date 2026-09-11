"""
config.py — Configuración central de Analizar-partido.

Repo standalone: pipeline de video -> eventos para partidos de Chaco For Ever.
No conoce nada de scraping de Sofascore ni de informes/redes sociales — eso vive
en el proyecto `Chaco For Ever Analisis`, que es quien CONSUME lo que exportamos
acá (events.csv). Ver PLAN.md y docs/DISENO_eventos.md.

Convención de carpetas (una por partido), bajo data/temporadas/{season}/partidos/{match_key}/:
    video/       el .mp4 del partido                         (lo pone el usuario)
    sofascore/   lineups_clean.csv, incidents_clean.csv,
                 statistics_clean.csv de ESE partido          (lo pone el usuario,
                                                                exportado del otro proyecto)
    tracking/    positions.csv, ball_positions.csv,
                 tracking_meta.json, possession.csv           (L2/L3 del pipeline)
    events/      events.csv, review_queue.csv, pass_view.csv  (L4/L5 del pipeline)
    outputs/     red de pases, video anotado, validación      (L6 del pipeline)

match_key: "YYYY-MM-DD_vs_<rival>_<h|a>" (misma convención que el otro proyecto).
"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Carga pares CLAVE=VALOR de .env en os.environ (sin pisar lo ya seteado).
    Dependency-free. .env está en .gitignore."""
    path = path or os.path.join(_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# ─── Temporada activa ────────────────────────────────────────────────────────

CURRENT_SEASON = "2026"

# ─── Rutas de datos ──────────────────────────────────────────────────────────

_SEASON_DATA_DIR = os.path.join(_ROOT, "data", "temporadas", "{season}", "partidos")


def season_from_match_key(match_key):
    """'2026-09-06_vs_estudiantes_caseros_h' -> '2026'. Fallback a CURRENT_SEASON."""
    prefix = (match_key or "")[:4]
    return prefix if prefix.isdigit() else CURRENT_SEASON


def season_data_dir(season=None):
    return _SEASON_DATA_DIR.format(season=season or CURRENT_SEASON)


def match_dir(match_key, season=None):
    """Carpeta raíz del partido: data/temporadas/{season}/partidos/{match_key}/"""
    s = season or season_from_match_key(match_key)
    return os.path.join(season_data_dir(s), match_key)


def match_video_dir(match_key, season=None):
    return os.path.join(match_dir(match_key, season), "video")


def match_sofascore_dir(match_key, season=None):
    """lineups_clean.csv / incidents_clean.csv / statistics_clean.csv de ese partido
    (exportados a mano del proyecto Chaco For Ever Analisis)."""
    return os.path.join(match_dir(match_key, season), "sofascore")


def match_tracking_dir(match_key, season=None):
    """Salida de L2/L3: positions.csv, ball_positions.csv, tracking_meta.json,
    possession.csv."""
    return os.path.join(match_dir(match_key, season), "tracking")


def match_events_dir(match_key, season=None):
    """Salida de L4/L5: events.csv, review_queue.csv, pass_view.csv."""
    return os.path.join(match_dir(match_key, season), "events")


def match_outputs_dir(match_key, season=None):
    """Salida de L6: red de pases, video anotado, reporte de validación."""
    return os.path.join(match_dir(match_key, season), "outputs")


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


def find_match_video(match_key, season=None):
    """Ruta del primer video encontrado en video/ del partido, o None."""
    vdir = match_video_dir(match_key, season)
    if not os.path.isdir(vdir):
        return None
    for f in sorted(os.listdir(vdir)):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            return os.path.join(vdir, f)
    return None


def find_match_sofascore(match_key, season=None, name="lineups_clean.csv"):
    """Ruta de un CSV de sofascore/ del partido (lineups_clean.csv por default),
    o None si no está."""
    p = os.path.join(match_sofascore_dir(match_key, season), name)
    return p if os.path.exists(p) else None


# ─── Pipeline de video ───────────────────────────────────────────────────────

MODELS_DIR = os.path.join(_ROOT, "models")

# Backend de modelos:
#   "roboflow" -> paquete `inference`, descarga los modelos con la API key (recomendado)
#   "local"    -> pesos .pt en MODELS_DIR (entrenados aparte, p.ej. tras fine-tuning F2)
VIDEO_MODEL_BACKEND = "roboflow"

# Modelos en Roboflow Universe (backend "roboflow"). El de jugadores ya incluye
# la pelota como clase 0; el de pelota dedicado es un refinamiento (F2).
ROBOFLOW_MODELS = {
    "player_detection": "football-players-detection-3zvbc/12",
    "pitch_detection":  "football-field-detection-f07vi/15",
    "ball_detection":   "football-ball-detection-rejhg/4",   # opcional, F2
}

# Pesos locales (backend "local", tras fine-tuning en F2).
VIDEO_MODELS = {
    "player_detection": os.path.join(MODELS_DIR, "football-player-detection.pt"),
    "pitch_detection":  os.path.join(MODELS_DIR, "football-pitch-detection.pt"),
    "ball_detection":   os.path.join(MODELS_DIR, "football-ball-detection.pt"),
}

# Frames por segundo a procesar (el video corre a ~25-30/50; muestrear baja el costo).
VIDEO_SAMPLE_FPS = 5


def roboflow_api_key():
    """API key de Roboflow desde la variable de entorno ROBOFLOW_API_KEY.
    En un notebook: os.environ['ROBOFLOW_API_KEY'] = 'tu_key' antes de procesar."""
    return os.environ.get("ROBOFLOW_API_KEY", "").strip()
