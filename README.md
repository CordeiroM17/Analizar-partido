# Analizar partido

Pipeline **video → flujo de eventos** para Chaco For Ever (Primera Nacional del fútbol
argentino), con herramientas gratis / open-source.

A partir del video de la transmisión de un partido y la ficha de Sofascore de ese mismo
partido, produce un `events.csv` canónico (pases, pérdidas, intercepciones, recuperaciones,
tiros, conducciones — quién interviene, dónde y cuándo) y, sobre eso, la **red de pases**.

- **Plan completo:** [`PLAN.md`](PLAN.md)
- **Diseño del esquema de eventos:** [`docs/DISENO_eventos.md`](docs/DISENO_eventos.md)
- **Ejemplos del formato que consume el proyecto de stats:** [`ejemplos csv otro proyeto/`](ejemplos%20csv%20otro%20proyeto/)

## Estado

- ✅ **F0** — repo standalone armado; `downloader.py` / `pipeline.py` / `schema.py` /
  `validation.py` portados y adaptados desde `Chaco For Ever Analisis` (que queda como
  consumidor del `events.csv`).
- 🔜 **F1** — partido de prueba listo (`data/temporadas/2026/partidos/2026-09-06_vs_estudiantes_caseros_h/`,
  video + CSV de Sofascore). Falta el timestamp del saque inicial para correr
  `notebooks/procesar_video_colab.ipynb` sobre los primeros 15 minutos.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt          # base (descarga, notebooks locales)
pip install -r requirements-video.txt    # solo en Colab/Kaggle (pipeline pesado)
```

## Restricciones de diseño

- Solo herramientas **gratis / open-source**.
- Corre en **Google Colab / Kaggle** (no en tiempo real).
- **Humano en el loop**: la máquina hace el 80–90%, el usuario corrige lo dudoso.
- **Precisión sobre exhaustividad**: se puede perder algún evento; nunca inventar uno.
- Coordenadas **OPTA / Sofascore 0–100** para enganchar con el ecosistema existente.
