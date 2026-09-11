# Plan — Analizar partido

Pipeline **video → flujo de eventos** para Chaco For Ever (Primera Nacional), con
herramientas gratis / open-source, corriendo en Colab/Kaggle, con **humano en el loop**
y criterio de **precisión sobre exhaustividad** (nunca inventar un evento).

> Planificado con el protocolo *grilling* (mattpocock/skills) el 2026-09-10.
> Este documento es la referencia completa. La vista visual está publicada como Artifact.

---

## 0. Qué es esto y cómo se conecta

`Analizar-partido` es un **repo standalone**. Toma el video de un partido + la ficha de
Sofascore de ese partido, y produce un `events.csv` canónico. El proyecto
`Chaco For Ever Analisis` (repo `Chaco.git`) queda como **consumidor** de ese CSV — ahí
viven los informes, las placas de Instagram y el modelo de la liga.

```
                 ┌─────────────────────────────┐
video LPF Play ─▶│      Analizar-partido        │─▶ events.csv ─┐
lineups_clean ──▶│  (visión + motor de eventos) │   pass_view.csv │
                 └─────────────────────────────┘   red de pases  │
                                                   video anotado │
                                                                 ▼
                                              Chaco For Ever Analisis
                                              (informes, redes, liga)
```

**Contrato de entrada** (una carpeta por partido):
- `data/temporadas/{season}/partidos/{match_key}/video/*.mp4`
- `data/temporadas/{season}/partidos/{match_key}/sofascore/lineups_clean.csv` (+ `incidents_clean.csv`, `statistics_clean.csv` para validar)
- `match_key`: `YYYY-MM-DD_vs_<rival>_<h|a>` (misma convención que el otro proyecto)

**Contrato de salida:**
- `.../{match_key}/events/events.csv` — tabla canónica (esquema en §6)
- `.../{match_key}/events/review_queue.csv` — eventos dudosos, para revisión manual
- `.../{match_key}/events/pass_view.csv` — subconjunto estilo SPADL para la red de pases
- `.../{match_key}/outputs/` — red de pases (PNG), video anotado, reporte de validación

**Sistema de coordenadas:** OPTA/Sofascore **0–100 × 0–100**, anisotrópico
(x sobre ~105 m, y sobre ~68 m). Para cualquier cálculo de distancia se pasa a metros
(`x·1.05`, `y·0.68`). Dirección de ataque normalizada por equipo y por tiempo.

---

## 1. Estado actual (lo que ya existe, a portar desde `Chaco.git`)

| Pieza | Archivo | Estado |
|---|---|---|
| Descarga de video | `downloader.py` + `descargar_video.ipynb` | ✅ Funciona. LPF Play = immergo.tv, **HLS sin DRM**; navegador headless captura el `master.m3u8`, `yt-dlp` baja. |
| Prototipo de tracking | `pipeline.py` + `procesar_video_colab.ipynb` | ⚠️ Prototipo sin validar. `roboflow/sports`: YOLOv8 (jugador/pelota/GK/árbitro) + ByteTrack + `TeamClassifier` (SigLIP+UMAP+KMeans) + keypoints de cancha + homografía → `positions.csv` + `ball_positions.csv` en OPTA 0–100. Corre en Colab T4 (el paquete `inference` no anda en Python 3.13 local). |
| Contrato de datos | `schema.py` | ✅ `positions.csv` + validadores. |
| Validación vs Sofascore | `validation.py` | ✅ `compare_with_lineups`, `players_per_frame`, `positions_plausibility`, `track_summary`. |
| Diseño de eventos | `docs/DISENO_eventos.md` | ✅ Plan F0–F6 + esquema `events.csv` general. F0 (esquema) congelado. |

**Problemas conocidos ya documentados:**
- Pelota detectada en **~74%** de los frames (huecos rompen la posesión). Hay un modelo dedicado `football-ball-detection` sin integrar.
- Tracking fragmentado: **~700 track_ids para 22 jugadores**, ~36% efímeros.
- El único video de prueba (`2026-04-19`) **no está en Sofascore** → sin validación numérica todavía.

**Tesis central del diseño:** el corazón del sistema no es "detectar pases" sino
**asignar posesión bien**. Cada evento es una transición en la línea de tiempo de posesión.

---

## 2. Arquitectura del pipeline (capas)

```
L0  Descarga            video del partido (.mp4)                      [✅ hecho]
      │
L1  Preparación         recorte a "juego en vivo" (PySceneDetect +    [nuevo]
      │                 ratio de verde + éxito de homografía +
      │                 OCR del marcador); anclaje manual del minuto
      ▼
L2  Tracking            detección + ByteTrack + equipos + homografía  [portar + endurecer]
      │                 → positions.csv, ball_positions.csv (OPTA)
      ▼
L3  Motor de posesión   pelota limpia (interpolar/suavizar) +         [F1 — nuevo]
      │                 poseedor por frame en metros → possession.csv
      ▼
L4  Identidad           SigLIP por track → clustering intra-equipo    [F3 — nuevo]
      │                 (restricción de coexistencia) → OCR de dorsal
      │                 (PARSeq, set cerrado del lineup) → etiquetado
      │                 humano de ~25-30 clusters → track_id → jugador
      ▼
L5  Segmentación        transiciones de posesión → pase / pérdida /   [F2, F4 — nuevo]
      │                 intercepción / recuperación / conducción /
      │                 tiro; confianza por evento
      ▼
L6  Enriquecimiento     longitud, dirección, progresivo, a último     [F5 — nuevo]
      │                 tercio, al área, termina en tiro/gol, subtipo
      ▼
L7  Revisión            notebook con widgets: etiquetado de identidad [nuevo]
      │                 + cola de dudosos (clip + actor/receptor
      │                 propuestos → aceptar / editar / rechazar)
      ▼
L8  Salidas             events.csv + pass_view.csv + red de pases +   [F6 — nuevo]
                        video anotado + reporte de validación
```

**Salida en dos niveles (decisión de este grilling):**
- `events.csv` = eventos de **alta confianza** + los **confirmados a mano**.
- `review_queue.csv` = todo lo dudoso (posesión ambigua, identidad incierta, pelota
  perdida en el tramo). Un evento **no entra** a `events.csv` sin superar el umbral o tu
  confirmación. Costo: más tiempo de revisión. Beneficio: casi cero eventos inventados.

---

## 3. Fases y hitos

Cada fase termina en un entregable verificable. Sin deadline: se avanza de a una.

### F0 — Setup del repo standalone ✅ (2026-09-10)
- Estructura de carpetas, `config.py` propio (sin nada de Sofascore), split de
  requirements (`requirements.txt` liviano/local, `requirements-video.txt` pesado/Colab),
  `.gitignore` (ignora `data/`, `models/`, `*.mp4`, `.venv/`).
- Portados y adaptados: `downloader.py`, `schema.py`, `pipeline.py` (+ `start_seconds`
  para anclar el saque inicial sin recortar el archivo), `validation.py`, el `DISENO` y
  los 3 notebooks (`descargar_video`, `analisis_video`, `procesar_video_colab`).
- **Hecho cuando:** `python -c "import config; from src.video import pipeline, schema, validation"` corre limpio y `descargar_video.ipynb` baja un video a la carpeta correcta. ✅ Verificado con un venv local (`pandas`/`numpy`); `find_match_video` encuentra el `.mp4` de `2026-09-06_vs_estudiantes_caseros_h`.

### F1 — Video de prueba + tracking base (medición)
- Elegir un partido de **local** que **esté en Sofascore**. Bajarlo. Traer su
  `lineups_clean.csv` / `incidents_clean.csv` / `statistics_clean.csv` del otro proyecto.
- Recortar los **primeros 15 min del primer tiempo**.
- Correr el pipeline portado en Colab/Kaggle (T4/P100, `SAMPLE_FPS=5`).
- Medir sobre el clip: tasa de detección, jugadores/frame vs 11, plausibilidad de
  homografía (% de puntos dentro de la cancha), fragmentación de tracks, cobertura de pelota.
- **Hecho cuando:** existe `positions.csv` + `ball_positions.csv` del clip + un reporte de
  métricas. **Compuerta de decisión:** ¿el tracking alcanza, o hay que fine-tunear (F2)?

### F2 — Endurecer el tracking *(condicional al resultado de F1)*
- Si las métricas flojean: etiquetar **300–800 frames** (CVAT / Roboflow free) tomados de
  **partidos históricos del Juan Alberto García** (DeporTV 2021–2025, TyC 2022–2025 en
  YouTube, descargables con `yt-dlp`); fine-tunear el detector y los keypoints de cancha.
- Integrar el modelo **dedicado de pelota** + interpolación de trayectoria + suavizado.
- Homografía: **suavizado temporal** (EMA/Kalman sobre H) + compuerta de confianza +
  "mantener la última buena" en frames malos.
- L1: segmentación de "juego en vivo" como preproceso (descartar repes, publicidad, primeros planos).
- **Hecho cuando:** las métricas de tracking del clip de 15 min superan el umbral acordado.

### F3 — Identidad (`track_id → jugador`) *(se puede solapar con F5)*
- SigLIP embedding por track → clustering **dentro de cada equipo** en ~14–18 identidades,
  con la restricción "dos tracks que coexisten en un frame no son el mismo jugador".
- Restringir el OCR de dorsal (**PARSeq**) al **set cerrado de números del lineup** de ese
  partido; voto temporal por cluster.
- En el notebook de revisión: un humano etiqueta **~25–30 clusters una vez** (se le muestra
  un recorte representativo → nombre desde `lineups_clean.csv`).
- Chaco For Ever: identidad **obligatoria**. Visitante: **oportunista** (si el OCR lee el
  dorsal con confianza, se mapea contra el lineup visitante; si no, queda "away #?").
- **Hecho cuando:** mapa `track_id → jugador` del clip; jugadores de Chaco nombrados.

### F4 — Eventos base + salida en dos niveles
- Transiciones de la línea de posesión → **pase / pérdida / intercepción / recuperación**.
  Confianza por evento. Pelota parada incluida y etiquetada (`pass_type`).
- `events.csv` (auto-alta + confirmados) / `review_queue.csv` (dudosos).
- Notebook de revisión: cola `needs_review` con clip corto + actor/receptor propuestos.
- Validar **totales** vs Sofascore: pases/equipo vs `statistics: Passes`, % de acierto vs
  `Accurate passes`, intercepciones, posesión %. **Objetivo: caer dentro de ±20–30%.**
- **Hecho cuando:** `events.csv` del clip de 15 min con **≥95% de precisión** en una
  muestra revisada y **0 pases inventados** en el set limpio.

### F5 — Enriquecimiento + subtipos
- Longitud, dirección, progresivo, a último tercio, al área, `leads_to_shot`/`leads_to_goal`,
  subtipos de pase (`short/long/through/cross/switch`).
- **Hecho cuando:** `events.csv` del clip lleva todas las columnas derivadas del esquema.

### F6 — Vista de pases + red + video anotado
- `pass_view.csv` (subconjunto estilo SPADL).
- Red de pases (`mplsoccer`): nodos en posición media, aristas = pases completos ≥ umbral,
  ventana hasta el primer cambio.
- Video anotado del clip: cajas + IDs + líneas de pase.
- Reporte de validación vs Sofascore.
- **Hecho cuando:** los 3 entregables existen para los 15 min y la red pasa la prueba de la vista.

### F7 — Escalar (HITO)
- Correr el pipeline completo sobre el **primer tiempo entero (45 min)**, después el partido
  completo.
- Presupuesto de revisión manual: **~2–4 h/partido** al principio, bajando con el tiempo.
- **Hito cumplido cuando:** para **un partido de local**, `events.csv` con ≥95% de precisión
  (0 inventados en el set limpio), red de pases del once inicial que pasa la prueba de la
  vista, y video anotado del primer tiempo.

### F8 — Generalizar *(post-hito, fuera de alcance por ahora)*
Partidos de visitante (cada cancha), identidad completa del rival (→ red de pases del rival,
scouting), tiros y conducciones (F4 del `DISENO`), otros tipos de evento, dashboard.

---

## 4. Plan de cómputo

| Momento | Plataforma | Notas |
|---|---|---|
| **Ahora (F0–F6)** | **Kaggle free** (P100 16 GB, ~30 h/semana, sesiones 12 h, corre en background con "Save & Run All") o **Colab free** (T4, se corta a los ~90 min de inactividad, no corre con la pestaña cerrada → procesar en trozos de 10–15 min con checkpoints a Drive) | Kaggle es más estable para lotes. |
| **Cuando el pipeline ~funcione** | **Colab Pro** US$9,99/mes (100 unidades ≈ 84 h de T4/mes) | El usuario paga cuando haga falta. Ejecución en segundo plano quizás requiera Pro+ — verificar. |
| **Storage** | **Google Drive (5 TB, ya disponible)** | Guardar frames como **archivos comprimidos (tar/zip), no imágenes sueltas** (cuotas de la API de Drive). Video completo 720p ≈ 3–4 GB. |

**Tiempos estimados de procesamiento:** clip de 15 min ≈ 15–40 min. Partido completo ≈
3–8 h en T4/P100 → siempre en trozos con checkpoint.

---

## 5. Checklist de tareas manuales (lo que hace el usuario)

**Una vez (si se dispara F2):**
- [ ] Etiquetar 300–800 frames de partidos históricos para el fine-tuning (CVAT / Roboflow).

**Por cada partido:**
- [ ] Bajar el video (`descargar_video.ipynb`) y subirlo a Drive.
- [ ] Traer `lineups_clean.csv` + `incidents_clean.csv` + `statistics_clean.csv` de ese partido.
- [ ] Marcar el frame del saque inicial (y el del segundo tiempo) para el anclaje de minuto.
- [ ] Etiquetar ~25–30 clusters de identidad una vez (recorte → nombre).
- [ ] Revisar la cola `review_queue.csv` (~2–4 h al principio): aceptar / editar / rechazar.

---

## 6. Esquema canónico `events.csv`

Del `docs/DISENO_eventos.md` (una fila por evento). El CSV de pases (`pass_view.csv`) es
la **vista de pases**: filtrar `type == 'pass'` y renombrar a columnas estilo SPADL.

| Columna | Descripción |
|---|---|
| `event_id` | id único |
| `seq_id` | id de la cadena de posesión (para secuencias pre-gol/pre-tiro) |
| `match_key`, `period`, `t_sec`, `minute` | cuándo |
| `type` | `pass` · `shot` · `interception` · `recovery` · `loss` · `carry` · `duel` · `clearance` |
| `subtype` | pase: `short/long/through/cross/switch` · tiro: `on_target/off/blocked/goal` |
| `team` | equipo del actor (`home` / `away`) |
| `actor_id`, `actor_name` | **quién hace** la acción (track → jugador vía `lineups_clean`) |
| `recipient_id`, `recipient_name` | receptor (pase completo) |
| `opponent_id`, `opponent_name` | quién interviene en contra (intercepción / duelo) |
| `start_x`, `start_y`, `end_x`, `end_y` | dónde (OPTA 0–100) |
| `zone` | respaldo robusto cuando la homografía es dudosa (tercio × canal) |
| `outcome` | `complete` · `incomplete` · `won` · `lost` |
| `pass_type` | `open_play` · `throw_in` · `corner` · `goal_kick` · `free_kick` · `kickoff` |
| `body_part` | `foot` · `head` · `other` · `unknown` |
| `length`, `direction`, `progressive`, `to_final_third`, `into_box` | derivados |
| `ball_speed`, `duration` | cinemática |
| `leads_to_shot`, `leads_to_goal` | la cadena terminó en peligro |
| `source` | `video_tracking` |
| `confidence` | confianza del detector (0–1) |
| `review_status` | `auto_high` · `human_confirmed` · `needs_review` |
| `src_track_actor`, `src_track_recipient` | track_ids internos (para auditar) |

**Validación** (no hay ground-truth pase-a-pase): (a) revisión manual de una muestra
aleatoria de los eventos emitidos — objetivo ≥95% correctos, 0 inventados; (b) cruce de
totales por jugador y por equipo contra los agregados de Sofascore (`total_passes`,
`accurate_passes`, `statistics_clean.csv`).

---

## 7. Decisiones cerradas (grilling 2026-09-10)

- **Uso no comercial** (redes + proyecto de stats + eventualmente cuerpo técnico) → licencias
  AGPL / CC-BY-NC son aceptables.
- **Precisión > exhaustividad.** Se puede perder algún pase en el medio; **nunca** inventar uno.
- **Humano en el loop** obligatorio: la máquina hace el 80–90%, el usuario corrige.
- **Prueba de concepto sobre 15 min de un partido de local**, después escalar.
- **`events.csv` del `DISENO` es canónico**; el de pases es una vista.
- **Coordenadas OPTA 0–100** para enganchar directo con el ecosistema Sofascore.
- **Identidad**: sembrado desde `lineups_clean.csv` (set cerrado de dorsales) + OCR + etiquetado
  manual una vez. Chaco obligatorio, visitante oportunista.
- **Fine-tuning**: primero medir con modelos pre-entrenados, después etiquetar dirigido.
- **Anclaje de minuto** manual (marcar saque inicial).
- **Repo standalone**; `Chaco For Ever Analisis` consume el `events.csv`.

---

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Una sola cámara → cobertura parcial del partido | Esperado. El flujo de eventos es parcial pero útil: la mayoría de los eventos relevantes pasan cerca de la pelota, que es lo que la cámara sigue. Se documenta como limitación. |
| Identidad fragmentada → el limitante nº1 para una red con nombres | Clustering + set cerrado de dorsales + etiquetado manual. Interino: identidades anónimas estables. |
| Pelota en jugadas rápidas → limitante nº1 de la precisión de pases | Modelo dedicado de pelota + interpolación + suavizado. |
| Sin ground-truth pase-a-pase | Validación por muestra revisada + cruce de totales con Sofascore (±20–30%). |
| LPF Play podría agregar DRM en el futuro | Mantener el `downloader` resiliente; YouTube histórico (DeporTV / TyC) como respaldo. |
| Colab free se corta / cuotas de Kaggle | Procesar en trozos con checkpoints a Drive; migrar a Colab Pro cuando el pipeline esté estable. |

---

## 9. Próximos pasos inmediatos

1. ~~**F0** — armar el esqueleto del repo y portar el código existente.~~ ✅ hecho.
2. ~~El usuario elige y baja un partido de local que esté en Sofascore, y trae sus CSV.~~ ✅
   `2026-09-06_vs_estudiantes_caseros_h` (Chaco 1–0 Estudiantes de Caseros, local, confirmado
   en Sofascore). Video (720p) + los 3 CSV ya están en la carpeta del partido.
3. **Pendiente del usuario:** el timestamp del saque inicial en el archivo de video
   (`START_SECONDS` en los notebooks) — la transmisión arranca antes del partido.
4. **F1** — correr `procesar_video_colab.ipynb` sobre los primeros 15 min (desde el
   saque inicial) y validar con `analisis_video.ipynb` (`positions_plausibility`,
   `players_per_frame`, `compare_with_lineups`). Ojo: hay una expulsión visitante al
   minuto 11 — desde ahí el clip es 11 vs 10, no 11 vs 11.
5. Con las métricas en la mano, decidir si F2 (fine-tuning) es necesario o se pasa directo a F3/F4.
