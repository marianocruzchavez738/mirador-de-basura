import streamlit as st

from computervision2 import tab_camara, tab_imagen, tab_testeo, tab_video
from computervision2.modelos import selector_modelos

# Configuración de página con estilo simple
st.set_page_config(page_title="Panel YOLO", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {font-weight: 400;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("Panel de inferencia YOLO")
st.caption("Selecciona uno o más modelos y elige un modo de entrada para ejecutar la detección.")
st.write("El mejor modelo es el basurav10 hasta ahora")

# Barra lateral: modelos (con el +) y confianza mínima, compartidos por todas las pestañas
modelos = selector_modelos()
conf = st.sidebar.slider("Confianza mínima", 0.05, 0.95, 0.25, 0.05)
st.sidebar.divider()

# Pestañas: cada una vive en su propio archivo dentro de partes/
tab_img, tab_vid, tab_cam, tab_test = st.tabs(["Imagen", "Video", "Cámara en vivo", "Testeo"])

with tab_img:
    tab_imagen.render(modelos, conf)
with tab_vid:
    tab_video.render(modelos, conf)
with tab_cam:
    tab_camara.render(modelos, conf)
with tab_test:
    tab_testeo.render(modelos, conf)
