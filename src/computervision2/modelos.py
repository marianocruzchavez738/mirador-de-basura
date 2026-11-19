"""Todo lo relacionado con elegir y cargar modelos YOLO."""
import re
from collections import namedtuple
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_MODELOS = 5

# Un color (BGR, el formato de OpenCV) por modelo, y su emoji equivalente para la barra lateral
COLORES = [(0, 200, 0), (0, 0, 255), (255, 120, 0), (0, 200, 255), (200, 0, 200)]
EMOJIS = ["🟢", "🔴", "🔵", "🟡", "🟣"]

# Un modelo ya cargado: su nombre, el objeto YOLO y el color con el que se dibuja
Modelo = namedtuple("Modelo", "nombre yolo color")


def _orden_natural(nombre):
    """'basurav10.pt' -> ['basurav', 10, '.pt']: así basurav2 va antes que basurav10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", nombre)]


def listar_modelos(directorio: Path = MODELS_DIR):
    """Devuelve la lista de archivos .pt de la carpeta models/, en orden natural."""
    if not directorio.exists():
        return []
    return sorted(
        (f.name for f in directorio.iterdir() if f.suffix.lower() == ".pt"),
        key=_orden_natural,
    )


@st.cache_resource
def _cargar(ruta: str):
    # cache_resource: el modelo se carga una vez y se reutiliza en cada rerun de Streamlit
    return YOLO(ruta)


def _mas():
    st.session_state.n_modelos = min(st.session_state.n_modelos + 1, MAX_MODELOS)


def _menos():
    st.session_state.n_modelos = max(st.session_state.n_modelos - 1, 1)


def selector_modelos():
    """Dibuja el selector en la barra lateral y devuelve la lista de Modelo elegidos."""
    st.sidebar.header("Modelo")

    disponibles = listar_modelos()
    if not disponibles:
        st.sidebar.warning("No se encontraron modelos en la carpeta 'models'.")
        st.sidebar.caption(f"Coloca archivos .pt en: {MODELS_DIR}")
        st.stop()

    st.session_state.setdefault("n_modelos", 1)

    # Un desplegable por cada modelo activo
    elegidos = []
    for i in range(st.session_state.n_modelos):
        nombre = st.sidebar.selectbox(
            f"{EMOJIS[i]} Modelo {i + 1}",
            disponibles,
            index=min(i, len(disponibles) - 1),
            key=f"modelo_{i}",
        )
        if nombre not in elegidos:  # si repites uno, se ignora
            elegidos.append(nombre)

    # Botones para añadir / quitar modelos
    col_mas, col_menos = st.sidebar.columns(2)
    col_mas.button("➕", on_click=_mas, disabled=st.session_state.n_modelos >= MAX_MODELOS,
                   use_container_width=True, help="Usar otro modelo a la vez")
    col_menos.button("➖", on_click=_menos, disabled=st.session_state.n_modelos <= 1,
                     use_container_width=True, help="Quitar el último modelo")

    modelos = []
    for i, nombre in enumerate(elegidos):
        try:
            yolo = _cargar(str(MODELS_DIR / nombre))
        except Exception as e:
            st.sidebar.error(f"Error al cargar {nombre}: {e}")
            st.stop()
        modelos.append(Modelo(nombre.removesuffix(".pt"), yolo, COLORES[i % len(COLORES)]))

    st.sidebar.caption("Activos: " + ", ".join(m.nombre for m in modelos))
    return modelos