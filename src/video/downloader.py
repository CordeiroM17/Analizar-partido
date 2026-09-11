"""
downloader.py (video)
=====================
Descarga el video de un partido a partir del link de la página del player.

Probado con lpfplay.com (Liga Profesional / AFA), que usa immergo.tv:
el stream es HLS plano (master.m3u8) SIN DRM. La URL del manifest no está
en el HTML — la pide el player por JavaScript — así que se captura con un
navegador headless leyendo los logs de red (misma técnica que el scraper).

Flujo:
  link de la página -> capture_stream_url() -> master.m3u8 -> yt-dlp -> video/

Requisitos: undetected_chromedriver (captura) y yt-dlp (descarga). ffmpeg
es opcional; el downloader nativo de yt-dlp baja el HLS a un solo archivo.
"""

import os
import json
import time

import config

# Patrones que delatan un manifest de streaming en el tráfico de red
_MANIFEST_HINTS = (".m3u8", ".mpd")
# Sufijos que NO son el video principal (subtítulos, thumbnails, i-frames)
_MANIFEST_SKIP = ("subs", "thumb", "iframe", "_vtt", "audio_only")


def capture_stream_url(page_url, headless=True, max_wait=40, referer=None,
                       verbose=True):
    """
    Abre la página del player en un navegador headless y captura la URL del
    manifest HLS/DASH desde los logs de red.

    Retorna la URL del master manifest, o None si no apareció.
    """
    import undetected_chromedriver as uc

    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--mute-audio")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    if verbose:
        print(f"Lanzando Chrome headless para {page_url} ...")
    driver = uc.Chrome(options=opts, enable_cdp_events=True)

    candidates = []
    try:
        driver.get(page_url)
        deadline = time.time() + max_wait
        while time.time() < deadline:
            time.sleep(2)
            try:
                logs = driver.get_log("performance")
            except Exception:
                logs = []
            for entry in logs:
                try:
                    msg = json.loads(entry["message"])["message"]
                except Exception:
                    continue
                if msg.get("method") not in ("Network.requestWillBeSent",
                                             "Network.responseReceived"):
                    continue
                params = msg.get("params", {})
                url = ((params.get("request") or {}).get("url")
                       or (params.get("response") or {}).get("url") or "")
                if not any(h in url for h in _MANIFEST_HINTS):
                    continue
                low = url.lower()
                if any(s in low for s in _MANIFEST_SKIP):
                    continue
                if url not in candidates:
                    candidates.append(url)
                    if verbose:
                        print(f"  manifest: {url}")
            # Preferimos un 'master' si ya apareció
            master = next((u for u in candidates if "master" in u.lower()), None)
            if master:
                return master
        # Sin master explícito: devolvemos el primer candidato
        return candidates[0] if candidates else None
    finally:
        driver.quit()


def _format_selector(quality):
    """
    Traduce una calidad amigable a un selector de formato de yt-dlp.
    Los HLS de immergo vienen muxeados (A+V juntos), así que filtramos por altura.
    """
    heights = {"360p": 360, "720p": 720, "1080p": 1080}
    h = heights.get(quality)
    if h is None:
        return "b"  # mejor disponible
    return f"b[height<={h}]/b"


def download_stream(manifest_url, out_path, quality="720p", referer=None,
                    test=False, verbose=True):
    """
    Descarga el stream HLS a out_path con yt-dlp (downloader nativo, sin ffmpeg).

    quality: '360p' | '720p' | '1080p' (720p recomendado para visión por computadora)
    test:    si True, baja solo un fragmento (para validar la cadena)
    """
    import yt_dlp

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # yt-dlp completa la extensión real según el contenedor
    tmpl = os.path.splitext(out_path)[0] + ".%(ext)s"

    ydl_opts = {
        "format": _format_selector(quality),
        "outtmpl": tmpl,
        "downloader": "native",
        "hls_use_mpegts": True,     # contenedor robusto para HLS sin ffmpeg
        "quiet": not verbose,
        "no_warnings": True,
        "noprogress": not verbose,
    }
    if referer:
        ydl_opts["http_headers"] = {"Referer": referer}
    if test:
        ydl_opts["test"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(manifest_url, download=True)
        final = ydl.prepare_filename(info)
    return final


def download_match_video(page_url, match_key, season=None, quality="720p",
                         referer="https://www.lpfplay.com/", headless=True,
                         test=False, verbose=True):
    """
    Pipeline completo: link de la página -> captura del manifest -> descarga
    en data/temporadas/{season}/partidos/{match_key}/video/.

    page_url puede ser el link de la página del player O directamente un .m3u8.
    Retorna la ruta del video descargado.
    """
    if any(page_url.endswith(h) for h in _MANIFEST_HINTS) or ".m3u8?" in page_url:
        manifest = page_url
        if verbose:
            print(f"Usando manifest directo: {manifest}")
    else:
        manifest = capture_stream_url(page_url, headless=headless,
                                      referer=referer, verbose=verbose)
        if not manifest:
            raise RuntimeError(
                "No se pudo capturar el manifest. Probá headless=False o "
                "revisá que el link sea de un partido disponible.")

    video_dir = config.match_video_dir(match_key, season)
    out_path = os.path.join(video_dir, f"{match_key}.mp4")
    if verbose:
        print(f"Descargando {quality} -> {out_path}")

    final = download_stream(manifest, out_path, quality=quality,
                            referer=referer, test=test, verbose=verbose)
    if verbose:
        size = os.path.getsize(final) / (1024 ** 2) if os.path.exists(final) else 0
        print(f"✅ Video guardado: {final}  ({size:.0f} MB)")
    return final
