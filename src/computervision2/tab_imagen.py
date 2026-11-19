"""Pestaña 'Imagen': una foto, todos los modelos activos."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from computervision2.inferencia import contar_por_modelo, detectar


def render(modelos, conf):
    st.subheader("Detección sobre imagen")
    archivo = st.file_uploader(
        "Seleccionar archivo de imagen", type=["jpg", "jpeg", "png"], key="img_uploader"
    )
    if archivo is None:
        return

    imagen = Image.open(archivo).convert("RGB")
    bgr = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)

    with st.spinner("Procesando..."):
        anotada, detecciones = detectar(modelos, bgr, conf)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        st.image(imagen, use_container_width=True)
    with col2:
        st.caption("Detección")
        st.image(anotada, channels="BGR", use_container_width=True)

    conteo = contar_por_modelo(detecciones, modelos)
    st.write("Objetos detectados: " + " | ".join(f"{n}: {c}" for n, c in conteo.items()))
    if detecciones:
        st.dataframe(pd.DataFrame(detecciones), use_container_width=True)

    _, buffer = cv2.imencode(".jpg", anotada)
    st.download_button(
        "Descargar imagen procesada", data=buffer.tobytes(),
        file_name="deteccion_imagen.jpg", mime="image/jpeg",
    )
