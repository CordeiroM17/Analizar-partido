"""
pipeline.py (video)
===================
Pipeline de visión por computadora: video del partido -> tracking en
coordenadas OPTA 0-100 (contrato de schema.py).

Basado en el stack open source roboflow/sports:
  - Detección jugadores/arqueros/árbitro/pelota (4 clases)
  - supervision ByteTrack           -> identidad persistente (track_id)
  - sports.TeamClassifier           -> cluster de camisetas (SigLIP+UMAP+KMeans)
  - sports.ViewTransformer          -> homografía (píxeles -> cancha real)

Backend de modelos (config.VIDEO_MODEL_BACKEND):
  - "roboflow": paquete `inference` + API key gratuita -> descarga automática
  - "local":    pesos .pt entrenados aparte en models/ (tras fine-tuning, F2)

ESTADO: prototipo Fase B / F1. Correr sobre un clip corto primero (ver
`start_seconds` / `max_seconds`) y validar con notebooks/analisis_video.ipynb
antes de procesar el partido entero.

`t_sec` en la salida es relativo a `start_seconds` (es decir, al saque inicial
si se pasa el anclaje de kickoff) — no al inicio del archivo de video. Así
`minute = t_sec // 60` es directamente el minuto de partido.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.video.schema import POSITIONS_COLUMNS, BALL_COLUMNS, save_tracking

# IDs de clase del modelo de detección de roboflow/sports
BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3

ROLE_BY_CLASS = {PLAYER_ID: "player", GOALKEEPER_ID: "goalkeeper",
                 REFEREE_ID: "referee"}


def _kp_confidence(kp):
    """supervision >=0.29 renombró KeyPoints.confidence -> keypoint_confidence
    (la vieja queda como alias deprecado que en 0.29.x ya devuelve None).
    Soporta ambas versiones."""
    conf = getattr(kp, "keypoint_confidence", None)
    if conf is None:
        conf = kp.confidence
    return conf

# Dimensiones de SoccerPitchConfiguration de roboflow/sports (centímetros)
PITCH_LENGTH_CM = 12000
PITCH_WIDTH_CM  = 7000


def check_environment(match_key=None, season=None):
    """
    Chequea todo lo necesario para correr el pipeline.
    Retorna dict {check: (ok, detalle)} para mostrar como checklist.
    """
    checks = {}
    backend = config.VIDEO_MODEL_BACKEND

    # Dependencias base
    base_pkgs = [("torch", "PyTorch"), ("supervision", "Supervision"),
                 ("sports", "Roboflow sports"), ("cv2", "OpenCV")]
    if backend == "roboflow":
        base_pkgs.append(("inference", "Roboflow inference"))
    else:
        base_pkgs.append(("ultralytics", "Ultralytics YOLO"))

    for pkg, label in base_pkgs:
        try:
            __import__(pkg)
            checks[label] = (True, "instalado")
        except ImportError:
            checks[label] = (False, "falta — ver requirements-video.txt")

    # GPU
    if checks.get("PyTorch", (False,))[0]:
        import torch
        if torch.cuda.is_available():
            checks["GPU"] = (True, torch.cuda.get_device_name(0))
        else:
            checks["GPU"] = (False, "sin CUDA — corre en CPU (lento)")

    # Modelos
    if backend == "roboflow":
        key = config.roboflow_api_key()
        checks["API key Roboflow"] = (
            bool(key),
            f"...{key[-4:]}" if key else "falta — os.environ['ROBOFLOW_API_KEY']='...'")
        for name, mid in config.ROBOFLOW_MODELS.items():
            if name == "ball_detection":
                continue  # opcional
            checks[f"modelo {name}"] = (True, f"{mid} (se descarga al correr)")
    else:
        for name, path in config.VIDEO_MODELS.items():
            if name == "ball_detection":
                continue
            ok = os.path.exists(path)
            checks[f"modelo {name}"] = (ok, path if ok else f"falta: {path}")

    # Video del partido
    if match_key:
        video = config.find_match_video(match_key, season)
        checks["video del partido"] = (
            video is not None,
            video or f"poner el video en {config.match_video_dir(match_key, season)}")

    return checks


def environment_ready(checks, require_video=True):
    skip = {"GPU"}
    if not require_video:
        skip.add("video del partido")
    return all(ok for name, (ok, _) in checks.items() if name not in skip)


class _DetectionModel:
    """
    Wrapper unificado de detección: expone .infer(frame) -> sv.Detections,
    funcione el backend con `inference` (Roboflow) o `ultralytics` (local).
    """
    def __init__(self, role, device):
        import supervision as sv
        self._sv = sv
        self.backend = config.VIDEO_MODEL_BACKEND
        if self.backend == "roboflow":
            from inference import get_model
            self._model = get_model(
                model_id=config.ROBOFLOW_MODELS[role],
                api_key=config.roboflow_api_key())
        else:
            from ultralytics import YOLO
            self._model = YOLO(config.VIDEO_MODELS[role]).to(device)

    def infer(self, frame, conf=0.3):
        sv = self._sv
        if self.backend == "roboflow":
            res = self._model.infer(frame, confidence=conf)[0]
            return sv.Detections.from_inference(res)
        res = self._model(frame, verbose=False)[0]
        return sv.Detections.from_ultralytics(res)

    def infer_keypoints(self, frame, conf=0.3):
        sv = self._sv
        if self.backend == "roboflow":
            res = self._model.infer(frame, confidence=conf)[0]
            return sv.KeyPoints.from_inference(res)
        res = self._model(frame, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(res)


def _fit_team_classifier(video_path, player_model, device, n_frames=30, stride_sec=20):
    """
    Junta recortes de jugadores a lo largo del video y ajusta el
    clasificador de equipos (clusters de camiseta).
    """
    import cv2
    import supervision as sv

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * stride_sec)

    crops = []
    for fidx in range(0, total, step):
        if len(crops) > n_frames * 15:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        det = player_model.infer(frame)
        det = det[det.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in det.xyxy]
    cap.release()

    if len(crops) < 20:
        raise RuntimeError(f"Solo {len(crops)} recortes de jugadores — "
                           "¿el modelo detecta algo en este video?")

    if config.TEAM_CLASSIFIER_BACKEND == "color":
        from src.video.team_color import ColorTeamClassifier
        classifier = ColorTeamClassifier()
    else:
        from sports.common.team import TeamClassifier
        classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    return classifier


def process_video(match_key, season=None, sample_fps=None, start_seconds=None,
                  max_seconds=None, device=None, save=True, progress_every=200):
    """
    Corre el pipeline completo sobre el video del partido.

    Parámetros
    ----------
    sample_fps    : frames por segundo a procesar (default config.VIDEO_SAMPLE_FPS)
    start_seconds : segundo del ARCHIVO en el que arrancar (p.ej. el saque inicial,
                    para saltarse la previa de la transmisión). None = desde el inicio.
    max_seconds   : procesar solo N segundos desde start_seconds (p.ej. 900 para un
                    clip de prueba de 15 min). None = hasta el final del video.
    device        : 'cuda' | 'cpu' (autodetecta si None)
    save          : guardar en tracking/ al terminar

    `t_sec` en la salida es relativo a `start_seconds` (0 = saque inicial si se pasó).

    Retorna (df_positions, df_ball, meta)
    """
    import cv2
    import torch
    import supervision as sv
    from sports.common.view import ViewTransformer
    from sports.configs.soccer import SoccerPitchConfiguration

    video_path = config.find_match_video(match_key, season)
    if not video_path:
        raise FileNotFoundError(
            f"No hay video en {config.match_video_dir(match_key, season)}")

    sample_fps = sample_fps or config.VIDEO_SAMPLE_FPS
    start_seconds = start_seconds or 0.0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if config.VIDEO_MODEL_BACKEND == "roboflow" and not config.roboflow_api_key():
        raise RuntimeError(
            "Falta la API key de Roboflow. En el notebook: "
            "os.environ['ROBOFLOW_API_KEY'] = 'tu_key'")

    print(f"Cargando modelos (backend={config.VIDEO_MODEL_BACKEND}, {device})...")
    player_model = _DetectionModel("player_detection", device)
    pitch_model  = _DetectionModel("pitch_detection", device)

    print("Ajustando clasificador de equipos...")
    team_classifier = _fit_team_classifier(video_path, player_model, device)

    pitch_config = SoccerPitchConfiguration()
    pitch_vertices = np.array(pitch_config.vertices)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_seconds * video_fps)
    end_frame = n_frames_total
    if max_seconds:
        end_frame = min(end_frame, start_frame + int(max_seconds * video_fps))
    stride = max(1, round(video_fps / sample_fps))

    print(f"Video: {os.path.basename(video_path)} | {video_fps:.0f} fps | "
          f"frames {start_frame}-{end_frame} de {n_frames_total} | "
          f"procesando 1 de cada {stride}")

    pos_rows, ball_rows = [], []
    processed = 0
    # Cache de equipo por track_id: el clasificador de equipos (SigLIP) es caro
    # y el equipo de un jugador no cambia durante su track -- clasificar una
    # vez por track en vez de en cada frame es el ahorro más grande del loop.
    team_by_track = {}

    for fidx in range(start_frame, end_frame, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = (fidx / video_fps) - start_seconds

        # 1. Detección + tracking
        det = player_model.infer(frame)
        people = det[det.class_id != BALL_ID].with_nms(threshold=0.5)
        people = tracker.update_with_detections(people)
        ball = det[det.class_id == BALL_ID]

        # 2. Homografía del frame (keypoints de la cancha)
        kp = pitch_model.infer_keypoints(frame)
        kp_conf = _kp_confidence(kp)
        if kp_conf is None or len(kp_conf) == 0:
            continue  # el modelo no detectó cancha en este frame (repetición, primer plano, etc.)
        mask = kp_conf[0] > 0.5
        if mask.sum() < 4:
            continue  # frame sin cancha suficiente
        transformer = ViewTransformer(
            source=kp.xy[0][mask].astype(np.float32),
            target=pitch_vertices[mask].astype(np.float32),
        )

        # 3. Equipos (clusters de camiseta sobre los jugadores de campo).
        # Solo se clasifica una vez por track_id (cacheado en team_by_track);
        # el equipo de un jugador no cambia mientras dura su track, y esto es
        # lo que más pesa del loop (una pasada de SigLIP por jugador nuevo).
        is_player = people.class_id == PLAYER_ID
        teams = np.full(len(people), -1)
        player_idx = np.where(is_player)[0]
        if len(player_idx) > 0:
            new_idx = [i for i in player_idx
                       if int(people.tracker_id[i]) not in team_by_track]
            if new_idx:
                crops = [sv.crop_image(frame, people.xyxy[i]) for i in new_idx]
                preds = team_classifier.predict(crops)
                for i, pred in zip(new_idx, preds):
                    team_by_track[int(people.tracker_id[i])] = int(pred)
            for i in player_idx:
                teams[i] = team_by_track[int(people.tracker_id[i])]

        # 4. Píxeles -> cancha (punto de apoyo = centro inferior del bbox)
        anchors = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(anchors.astype(np.float32))

        for i in range(len(people)):
            pos_rows.append({
                "frame": fidx,
                "t_sec": round(t_sec, 2),
                "track_id": int(people.tracker_id[i]),
                "role": ROLE_BY_CLASS.get(int(people.class_id[i]), "player"),
                "team": int(teams[i]),
                "x": round(float(pitch_xy[i][0]) / PITCH_LENGTH_CM * 100, 2),
                "y": round(float(pitch_xy[i][1]) / PITCH_WIDTH_CM * 100, 2),
                "confidence": round(float(people.confidence[i]), 3),
            })

        if len(ball) > 0:
            b_anchor = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            b_xy = transformer.transform_points(b_anchor.astype(np.float32))
            ball_rows.append({
                "frame": fidx,
                "t_sec": round(t_sec, 2),
                "x": round(float(b_xy[0][0]) / PITCH_LENGTH_CM * 100, 2),
                "y": round(float(b_xy[0][1]) / PITCH_WIDTH_CM * 100, 2),
                "confidence": round(float(ball.confidence[0]), 3),
            })

        processed += 1
        if processed % progress_every == 0:
            print(f"  frame {fidx}/{end_frame} ({t_sec/60:.1f} min de partido)")

    cap.release()

    df_positions = pd.DataFrame(pos_rows, columns=POSITIONS_COLUMNS)
    df_ball = pd.DataFrame(ball_rows, columns=BALL_COLUMNS)

    meta = {
        "match_key": match_key,
        "video": os.path.basename(video_path),
        "video_fps": video_fps,
        "sample_fps": sample_fps,
        "start_seconds": start_seconds,
        "max_seconds": max_seconds,
        "frames_procesados": processed,
        "device": device,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "backend": config.VIDEO_MODEL_BACKEND,
        "modelos": (config.ROBOFLOW_MODELS if config.VIDEO_MODEL_BACKEND == "roboflow"
                    else {k: os.path.basename(v) for k, v in config.VIDEO_MODELS.items()}),
    }

    if save and not df_positions.empty:
        out = save_tracking(df_positions, df_ball, meta, match_key, season)
        print(f"✅ Tracking guardado en {out}")

    return df_positions, df_ball, meta
