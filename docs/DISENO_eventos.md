# Diseño: detección de EVENTOS desde el tracking de video

**Objetivo:** convertir el tracking (posiciones de jugadores + pelota por frame) en un
**flujo de eventos** del partido — no solo pases, sino tiros, intercepciones, pérdidas,
recuperaciones, conducciones, duelos. De cada acción queremos saber **quién interviene,
dónde y cuándo**. La red de pases es UNA vista sobre ese flujo, no el fin.

> Estado: diseño/ideación (2026-06-23). Portado a `Analizar-partido` el 2026-09-10 como
> documento canónico del esquema de eventos. El tracking ya está validado en la PC del
> proyecto original. Esto define la capa que va ARRIBA del tracking.

> **Decisiones (2026-06-23):**
> - **Validación SÍ es posible:** en la temporada 2026 vamos a tener **video + datos de
>   SofaScore** del mismo partido. Eso habilita cruzar los totales de eventos
>   (pases/tiros/intercepciones) contra SofaScore. El motor se valida de verdad.
> - **Arranque de implementación: F1 — Motor de posesión.** Es el cimiento del que sale todo.

---

## 1. La idea central: el motor de posesión

**Todo evento sale de una sola pregunta: ¿quién tiene la pelota en cada instante?**

Si sabemos, frame a frame, qué jugador controla la pelota (o si está "viajando"),
entonces los eventos son **transiciones** en esa línea de tiempo de posesión:

```
posesión A (Chaco #5) ──vuelo──> posesión B (Chaco #8)        = PASE completo (5→8)
posesión A (Chaco #5) ──vuelo──> posesión B (Rival #4)        = PASE perdido + INTERCEPCIÓN de #4
posesión A (Chaco #9) ──vuelo hacia el arco──> arquero/afuera = TIRO de #9
posesión A (Chaco #5) se mueve 15m con la pelota             = CONDUCCIÓN de #5
pelota suelta, dos rivales cerca, cambia de dueño            = DUELO / RECUPERACIÓN
```

Entonces el corazón del sistema NO es "detectar pases" sino **asignar posesión bien**.
Una vez que eso está, cada tipo de evento es un detector chico sobre la misma base.

---

## 2. Las capas del pipeline (de tracking a eventos)

```
positions.csv + ball_positions.csv   (ya lo tenemos)
        │
        ▼
[L1] Trayectoria de pelota limpia    → interpolar huecos (hoy 74% de frames con pelota),
        │                              suavizar, velocidad y dirección por frame,
        │                              estado: "controlada" vs "en vuelo"
        ▼
[L2] Posesión por frame              → jugador más cercano a la pelota (en METROS, no OPTA),
        │                              dentro de un radio y con la pelota "pegada".
        │                              Salida: línea de tiempo (t, poseedor_track, equipo)
        ▼
[L3] Segmentación en eventos         → spells de posesión + transiciones →
        │                              pase / pérdida / intercepción / tiro / conducción / duelo
        ▼
[L4] Enriquecimiento                 → quién (nombre via identidad), dónde (x,y inicio/fin),
        │                              cuándo (min), longitud, dirección, progresivo, a último
        │                              tercio, al área, ¿termina en tiro/gol?, subtipo
        ▼
events.csv                           → la tabla canónica (sección 3)
```

Detalle importante de L2: las coordenadas OPTA son **anisotrópicas** (x: 0-100 sobre
~105 m, y: 0-100 sobre ~68 m). Para medir "cercanía a la pelota" hay que pasar a metros
(x·1.05, y·0.68); si no, la distancia está deformada.

---

## 3. Esquema canónico `events.csv` (general, no solo pases)

Una fila por evento. Reemplaza y generaliza el `pass_events.csv` que se había bosquejado.

| Columna | Descripción |
|---|---|
| `event_id` | id único |
| `seq_id` | id de la cadena de posesión (para secuencias pre-gol/pre-tiro) |
| `match_key`, `period`, `t_sec`, `minute` | cuándo |
| `type` | `pass` · `shot` · `interception` · `recovery` · `loss` · `carry` · `duel` · `clearance` |
| `subtype` | pase: `short/long/through/cross/switch` · tiro: `on_target/off/blocked/goal` |
| `team` | equipo del actor |
| `actor_id`, `actor_name` | **quién hace** la acción (track→jugador) |
| `recipient_id`, `recipient_name` | receptor (pase completo) |
| `opponent_id`, `opponent_name` | **quién interviene en contra** (intercepción/duelo) |
| `start_x`, `start_y` | dónde empieza (OPTA) |
| `end_x`, `end_y` | dónde termina (destino del pase/tiro/conducción) |
| `outcome` | `complete` · `incomplete` · `won` · `lost` |
| `length`, `direction`, `progressive`, `to_final_third`, `into_box` | atributos derivados |
| `ball_speed`, `duration` | cinemática |
| `leads_to_shot`, `leads_to_goal` | la cadena terminó en peligro |
| `source` | `video_tracking` (vs `sofascore`) |
| `confidence` | confianza del detector (para filtrar después) |

Con `actor` + `recipient` + `opponent` + `start/end` + `type`, una sola tabla responde
**todo** lo que pediste: quiénes intervienen, dónde y cuándo, en cualquier tipo de acción.

> **Extensión (grilling 2026-09-10):** se agregan `zone` (respaldo cuando la homografía es
> dudosa), `pass_type` (`open_play/throw_in/corner/goal_kick/free_kick/kickoff`), `body_part`,
> `review_status` (`auto_high/human_confirmed/needs_review`) y `src_track_*` para auditar.

---

## 4. Las DOS dependencias duras (honestidad)

El motor de posesión es factible con lo que tenemos. Pero hay dos cuellos de botella
reales que definen hasta dónde llega la calidad:

### 4.1 Identidad: `track_id → jugador` (lo que da los NOMBRES)
- Hoy el tracking se fragmenta (~36% de tracks efímeros, ~700 ids para 22 jugadores).
  Para un PASE individual (dura ~1 s) el track del que da y del que recibe suele
  sobrevivir, así que el **evento** se puede detectar igual. Pero para poner **nombres**
  y para **encadenar** una red por jugador, hace falta colapsar los fragmentos.
- **Plan: re-identificación + etiquetado una vez.**
  1. Embeber cada track con su apariencia (SigLIP — el mismo modelo que ya usamos para
     los equipos) → un vector promedio por track.
  2. Clusterizar los tracks DENTRO de cada equipo en ~14-18 identidades (con la
     restricción "dos tracks que coexisten en un frame no son el mismo jugador").
  3. Un humano etiqueta ~25-30 clusters una sola vez por partido (se le muestra un
     recorte representativo) → nombre. Ahí `actor_name` deja de ser anónimo.
- **Interino sin nombres:** se puede armar todo con identidades anónimas estables
  ("jugador C0-07") y etiquetar después; la estructura (duplas, zonas) ya sirve.

### 4.2 Calidad de la pelota (lo que da la PRECISIÓN del evento)
- Hoy la pelota aparece en 74% de los frames. Los huecos rompen la posesión.
- **Plan:** usar el modelo dedicado `football-ball-detection.pt` + interpolar la
  trayectoria + suavizar. La pelota es la clase más difícil; este es el factor que más
  mueve la aguja en la precisión de pases/tiros.

---

## 5. Qué es factible YA / qué necesita identidad / qué es marginal

| Nivel | Qué sale | Depende de |
|---|---|---|
| ✅ Factible con lo actual | dónde y cuándo de cada acción; pase vs pérdida; ubicación de intercepciones, tiros, recuperaciones; mapas por equipo; cadenas que terminan en tiro | motor de posesión + pelota |
| 🔸 Necesita identidad | quién da/recibe (nombres); red de pases por jugador; duplas; sonar por jugador | re-ID + etiquetado (4.1) |
| ⚠️ Marginal / ruidoso | subtipos finos (pase filtrado vs normal, duelo vs intercepción), conducciones cortas | heurísticas + más datos |

Mensaje claro: **el "dónde/cuándo/qué" de las acciones es alcanzable pronto**; el "quién"
con nombre exige el paso de identidad; los matices finos quedan para el final.

---

## 6. Validación (contra SofaScore)

El flujo de eventos se valida cruzando TOTALES contra SofaScore en un partido que esté
en SofaScore:
- nº de pases por equipo  vs `statistics: Passes`
- nº de tiros             vs `shotmap` (cantidad)
- intercepciones          vs `statistics: Interceptions`
- % de acierto de pases   vs `Accurate passes`

Si caemos dentro de ~±20-30%, el motor es creíble. **Para validar de verdad necesitamos
el video de un partido que sí esté en SofaScore.** Hasta entonces, solo validación visual
(face-validity).

---

## 7. Gráficas posibles (una vez existe `events.csv`)

Todas salen de la misma tabla, filtrando por `type`/`team`/`player`:
- **Red de pases** (nodos = jugadores en su posición media, aristas = pases completos).
- **Mapa de intercepciones / recuperaciones** — dónde roba cada equipo.
- **Mapa de pérdidas** — dónde se pierde la pelota.
- **Mapa de tiros con contexto** (la jugada previa, no solo el remate).
- **Cadenas pre-gol / pre-tiro** — las asociaciones que más se repiten antes del peligro.
- **Sonar de pases por jugador** — direcciones y longitudes típicas.
- **Redes por tercios / por fase del partido**.
- **"Quiénes intervienen"** — para cualquier acción, actor + receptor/oponente.

---

## 8. Plan por fases (ver `PLAN.md` para la versión vigente)

- [x] **F0 — Congelar el esquema `events.csv`** (sección 3). Cimiento de todo.
- [ ] **F1 — Motor de posesión** (L1+L2): pelota limpia + poseedor por frame, en metros.

  **Spec concreta de F1:**
  - *Entrada:* `tracking/positions.csv` + `tracking/ball_positions.csv`.
  - *L1 — pelota limpia:* reindexar la pelota a todos los frames procesados; interpolar
    huecos cortos (≤ ~0.5 s) linealmente, marcar los largos como "sin pelota"; suavizar
    (media móvil corta); calcular `ball_speed` y `ball_dir` por frame. (Opcional: sumar
    el modelo dedicado `football-ball-detection.pt`.)
  - *L2 — poseedor por frame:* pasar todo a **metros** (x·1.05, y·0.68); poseedor =
    jugador más cercano a la pelota con `dist < R` (R≈2 m) **y** `ball_speed` baja
    (pelota controlada, no en vuelo). Si nadie cumple → `loose`/`flight`.
    Suavizar la línea (un poseedor debe durar ≥ N frames para contar; tolerar 1-2 frames
    de hueco).
  - *Salida intermedia:* `tracking/possession.csv` con
    `frame, t_sec, state(controlada/vuelo/suelta), possessor_track, team, ball_x, ball_y,
    ball_speed`.
  - *Validación F1:* pintar el poseedor resaltado y ver que "sigue" a quien tiene la
    pelota; y que el % de tiempo de posesión por equipo se parezca al `Ball possession`
    de SofaScore.
- [ ] **F2 — Eventos base** (L3): pase completo, pérdida, intercepción, recuperación.
      Validar TOTALES vs SofaScore.
- [ ] **F3 — Identidad** (4.1): re-ID por apariencia + etiquetado manual → nombres.
- [ ] **F4 — Tiros y conducciones** (L3+): lógica de dirección al arco + cross-check shotmap.
- [ ] **F5 — Enriquecimiento + subtipos** (L4): longitud, progresivo, área, cadenas a gol.
- [ ] **F6 — Gráficas** (sección 7), arrancando por red de pases + mapa de intercepciones.

**Orden recomendado:** F0 → F1 → F2 (con validación) → F3 (nombres) → F4/F5 → F6.
La identidad (F3) se puede empezar en paralelo a F2 porque es independiente.

---

## 9. Riesgo principal

Una sola cámara de TV (pans, cortes, cobertura parcial) hace que el flujo de eventos sea
**parcial**: vamos a capturar bien las acciones cerca de la pelota (que es lo que la
cámara sigue), pero no el 100% del partido. Es esperable y útil igual — la mayoría de los
eventos relevantes pasan cerca de la pelota.
