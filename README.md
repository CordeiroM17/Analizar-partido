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

Planificado (grilling, 2026-09-10). Repo recién separado del proyecto `Chaco For Ever
Analisis`, que queda como consumidor del `events.csv`.

Próximo paso: **F0** — armar el esqueleto del repo y portar el código de descarga y
tracking que ya existe.

## Restricciones de diseño

- Solo herramientas **gratis / open-source**.
- Corre en **Google Colab / Kaggle** (no en tiempo real).
- **Humano en el loop**: la máquina hace el 80–90%, el usuario corrige lo dudoso.
- **Precisión sobre exhaustividad**: se puede perder algún evento; nunca inventar uno.
- Coordenadas **OPTA / Sofascore 0–100** para enganchar con el ecosistema existente.
