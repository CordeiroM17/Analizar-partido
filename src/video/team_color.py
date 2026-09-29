"""
team_color.py
==============
Clasificador de equipos rápido por color de camiseta (histograma HSV +
KMeans), en vez de embeddings de un modelo pesado (SigLIP).

Por qué: en broadcast con cámara de TV, ByteTrack fragmenta mucho los tracks
(en una prueba real, ~300 track_ids en 1 minuto para 22 jugadores). Como el
equipo se re-clasifica en cada track "nuevo", con SigLIP eso significa correr
un modelo de visión pesado cientos de veces por minuto — es el cuello de
botella nº1 del pipeline, muy por encima de los dos modelos de Roboflow.
Un histograma de color es miles de veces más barato y alcanza para separar
dos camisetas visualmente distintas.

Misma interfaz que `sports.common.team.TeamClassifier` (`.fit(crops)` /
`.predict(crops)`), pensada como reemplazo directo.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


def _torso_descriptor(crop, h_bins=12, s_bins=4, v_bins=4):
    """Histograma HSV normalizado de la franja de torso de un recorte de
    jugador (evita el pasto de abajo y la piel/pelo de arriba)."""
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return np.zeros(h_bins * s_bins * v_bins, dtype=np.float32)
    torso = crop[int(h * 0.15):int(h * 0.55), int(w * 0.15):int(w * 0.85)]
    if torso.size == 0:
        torso = crop
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [h_bins, s_bins, v_bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


class ColorTeamClassifier:
    """Clasificador de equipos por color de camiseta. `n_clusters=2` separa
    los dos equipos; el árbitro/arquero quedan mezclados con lo más parecido
    (no hay clase aparte -- eso se resuelve más adelante con el rol de la
    detección, no acá)."""

    def __init__(self, n_clusters=2, **kwargs):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, crops):
        if not crops:
            raise RuntimeError("ColorTeamClassifier.fit: no llegaron recortes")
        feats = np.stack([_torso_descriptor(c) for c in crops])
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
        self.kmeans.fit(feats)
        return self

    def predict(self, crops):
        if self.kmeans is None:
            raise RuntimeError("ColorTeamClassifier: llamar a fit() antes de predict()")
        if not crops:
            return np.array([], dtype=int)
        feats = np.stack([_torso_descriptor(c) for c in crops])
        return self.kmeans.predict(feats)
