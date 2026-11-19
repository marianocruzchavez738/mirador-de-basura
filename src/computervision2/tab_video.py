"""Pestaña 'Video': procesa un video fotograma a fotograma con todos los modelos activos."""
import os
import tempfile

import cv2
import streamlit as st

from computervision2.inferencia import detectar


def _procesar(archivo, modelos, conf):
    """Procesa el video y devuelve los bytes del mp4 resultante."""
    entrada = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    entrada.write(archivo.getvalue())
    entrada.close()

    cap = cv2.VideoCapture(entrada.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.caption(f"Resolución: {ancho}x{alto} | FPS: {fps:.2f} | Fotogramas: {total}")

    salida_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(salida_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))

    barra = st.progress(0, text="Procesando video...")
    n = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        anotado, _ = detectar(modelos, frame, conf)
        out.write(anotado)
        n += 1
        if total > 0:
            barra.progress(min(n / total, 1.0))

    cap.release()
    out.release()
    barra.empty()

    with open(salida_path, "rb") as f:
        datos = f.read()
    os.unlink(entrada.name)
    os.unlink(salida_path)
    return datos, n


def render(modelos, conf):
    st.subheader("Detección sobre video")
    archivo = st.file_uploader(
        "Seleccionar archivo de video", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader"
    )
    if archivo is None:
        return

    # El resultado se guarda para no reprocesar todo el video cuando Streamlit
    # vuelve a ejecutar el script (por ejemplo, al pulsar "Descargar").
    clave = (archivo.name, archivo.size, tuple(m.nombre for m in modelos), conf)
    if st.session_state.get("video_clave") != clave:
        datos, n = _procesar(archivo, modelos, conf)
        st.session_state.video_clave = clave
        st.session_state.video_datos = datos
        st.session_state.video_frames = n

    st.write(f"Procesamiento completado. Fotogramas procesados: {st.session_state.video_frames}")
    st.video(st.session_state.video_datos)
    st.download_button(
        "Descargar video procesado", data=st.session_state.video_datos,
        file_name="deteccion_video.mp4", mime="video/mp4",
    )
