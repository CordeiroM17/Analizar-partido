"""
homography.py
==============
Homografía robusta a outliers para el pipeline de tracking.

Por qué existe: `sports.common.view.ViewTransformer` (roboflow/sports) calcula
la homografía con `cv2.findHomography(source, target)` SIN método robusto
(mínimos cuadrados simple). Si UN SOLO keypoint de cancha está mal ubicado
(línea borrosa, oclusión, falso positivo del modelo), arruina la homografía
del frame entero. Medido en F1: 4,1% de las posiciones con valores
catastróficamente fuera de rango OPTA (hasta x=-4274, y=1925 sobre una
cancha 0-100) -- justo el síntoma de esto.

RANSAC tolera que 1-2 keypoints de los ~6-32 detectados estén mal: encuentra
el subconjunto de puntos que sí son consistentes entre sí y calcula la
homografía solo con esos ("inliers"), descartando los que no encajan.
"""

import cv2
import numpy as np


class RobustViewTransformer:
    """Mismo uso que sports.common.view.ViewTransformer (`.transform_points(pts)`),
    pero con `cv2.findHomography(..., method=cv2.RANSAC)`. Expone `n_inliers` para
    que el llamador pueda descartar el frame si muy pocos puntos fueron consistentes
    (compuerta de confianza), en vez de usar una homografía calculada sobre solo 1-2
    puntos "buenos" de los 4+ detectados.
    """

    def __init__(self, source, target, reproj_threshold=8.0):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if source.shape != target.shape or source.shape[1] != 2:
            raise ValueError("source y target deben ser Nx2 y del mismo shape")

        self.m = None
        self.n_inliers = 0
        if len(source) < 4:
            return  # cv2.findHomography necesita al menos 4 puntos

        m, mask = cv2.findHomography(source, target, cv2.RANSAC, reproj_threshold)
        if m is None:
            return
        self.m = m
        self.n_inliers = int(mask.sum()) if mask is not None else len(source)

    @property
    def ok(self):
        return self.m is not None

    def transform_points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points
        if self.m is None:
            raise ValueError("Homografía no calculada (ver .ok antes de llamar)")
        reshaped = points.reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(reshaped, self.m)
        return out.reshape(-1, 2).astype(np.float32)
