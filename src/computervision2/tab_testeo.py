"""Pestaña 'Testeo': lote temporal de fotos, tabla de estadísticas, revisión una por una y guardado por grupos."""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from computervision2.inferencia import detectar, resumen_texto

BASE = Path(__file__).resolve().parent.parent
DIR_TEST = BASE / "test"            # aquí se guardan las fotos originales
DIR_RESULTADOS = BASE / "resultados"  # aquí se guardan las fotos con detección y los CSV
EXTENSIONES = ["jpg", "jpeg", "png", "bmp", "webp"]


# ---------- utilidades ----------

def _limpiar_nombre(texto):
    """Deja solo letras, números, guiones y espacios (para que sea un nombre de carpeta válido)."""
    return re.sub(r"[^\w\- ]", "", texto).strip()


def _grupos():
    """Carpetas (grupos) que ya existen en test/ o resultados/."""
    nombres = set()
    for d in (DIR_TEST, DIR_RESULTADOS):
        if d.exists():
            nombres |= {p.name for p in d.iterdir() if p.is_dir()}
    return sorted(nombres)


def _nombre_libre(nombre, existentes):
    """Si 'foto.jpg' ya está en el lote, devuelve 'foto_1.jpg', 'foto_2.jpg'..."""
    if nombre not in existentes:
        return nombre
    p, i = Path(nombre), 1
    while f"{p.stem}_{i}{p.suffix}" in existentes:
        i += 1
    return f"{p.stem}_{i}{p.suffix}"


def _decodificar(datos):
    return cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)


# ---------- pestaña ----------

def render(modelos, conf):
    ss = st.session_state
    ss.setdefault("test_fotos", {})        # nombre -> bytes de la foto (lote temporal)
    ss.setdefault("test_cache", {})        # resultados ya calculados, para no repetir inferencia
    ss.setdefault("test_uploader_n", 0)    # cambia la clave del uploader para vaciarlo tras añadir

    st.subheader("Testeo por lotes")

    # 1) Añadir fotos al lote temporal
    nuevas = st.file_uploader(
        "Selecciona fotos y pulsa 'Añadir al lote' (puedes repetir varias veces)",
        type=EXTENSIONES, accept_multiple_files=True, key=f"test_uploader_{ss.test_uploader_n}",
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Añadir al lote", disabled=not nuevas):
        for f in nuevas:
            ss.test_fotos[_nombre_libre(f.name, ss.test_fotos)] = f.getvalue()
        ss.test_uploader_n += 1
        st.rerun()
    if c2.button("Vaciar lote", disabled=not ss.test_fotos):
        ss.test_fotos, ss.test_cache = {}, {}
        st.rerun()

    st.caption(f"Fotos en el lote: {len(ss.test_fotos)} (temporales, no se guardan hasta que pulses 'Guardar')")
    if not ss.test_fotos:
        return

    # 2) Analizar (solo lo que falte; se descarta lo calculado con otros modelos/confianza)
    clave = (tuple(m.nombre for m in modelos), conf)
    ss.test_cache = {k: v for k, v in ss.test_cache.items() if k[1:] == clave}
    pendientes = [n for n in ss.test_fotos if (n,) + clave not in ss.test_cache]
    if pendientes:
        barra = st.progress(0, text="Analizando fotos...")
        for i, n in enumerate(pendientes, 1):
            ss.test_cache[(n,) + clave] = detectar(modelos, _decodificar(ss.test_fotos[n]), conf)
            barra.progress(i / len(pendientes), text=f"Analizando fotos... {i}/{len(pendientes)}")
        barra.empty()

    # 3) Tabla de estadísticas
    filas = []
    for n in ss.test_fotos:
        _, dets = ss.test_cache[(n,) + clave]
        fila = {"imagen": n, "hay_basura": "SI" if dets else "NO"}
        for m in modelos:
            fila[m.nombre] = sum(1 for d in dets if d["modelo"] == m.nombre)
        fila["detecciones"] = resumen_texto(dets)
        filas.append(fila)
    tabla = pd.DataFrame(filas)

    comparacion = pd.DataFrame(
        [{"modelo": m.nombre, "imagenes": len(tabla), "con_basura": int((tabla[m.nombre] > 0).sum())}
         for m in modelos]
    )
    st.markdown("#### Estadísticas")
    st.caption(f"Imágenes con basura detectada por algún modelo: "
               f"{int((tabla['hay_basura'] == 'SI').sum())} de {len(tabla)}")
    st.dataframe(comparacion, use_container_width=True, hide_index=True)
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    # 4) Revisar una por una
    st.markdown("#### Revisar foto por foto")
    elegida = st.selectbox("Foto", list(ss.test_fotos), key="test_ver")
    anotada, dets = ss.test_cache[(elegida,) + clave]
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        st.image(_decodificar(ss.test_fotos[elegida]), channels="BGR", use_container_width=True)
    with col2:
        st.caption("Detección")
        st.image(anotada, channels="BGR", use_container_width=True)
    if dets:
        st.dataframe(pd.DataFrame(dets), use_container_width=True, hide_index=True)
    else:
        st.caption("Ningún modelo detectó nada en esta foto.")

    # 5) Carpetas (grupos) y guardado
    st.markdown("#### Guardar")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    nueva = c1.text_input("Nueva carpeta (grupo)", key="test_nueva", placeholder="ej: calle_lima_1")
    if c2.button("Crear carpeta", use_container_width=True):
        nombre = _limpiar_nombre(nueva)
        if not nombre:
            st.warning("Escribe un nombre válido para la carpeta.")
        else:
            (DIR_TEST / nombre).mkdir(parents=True, exist_ok=True)
            (DIR_RESULTADOS / nombre).mkdir(parents=True, exist_ok=True)
            ss.test_grupo = nombre  # queda seleccionada como destino
            st.rerun()

    grupos = _grupos()
    if not grupos:
        st.info("Crea una carpeta para poder guardar el lote.")
        return
    grupo = st.selectbox("Guardar en la carpeta", grupos, key="test_grupo")

    d_test, d_res = DIR_TEST / grupo, DIR_RESULTADOS / grupo
    repetidas = [n for n in ss.test_fotos if (d_test / n).exists()]
    if repetidas:
        st.warning(f"{len(repetidas)} foto(s) ya existen en '{grupo}' y se sobrescribirán.")

    if st.button("Guardar lote", type="primary"):
        d_test.mkdir(parents=True, exist_ok=True)
        d_res.mkdir(parents=True, exist_ok=True)
        for n in ss.test_fotos:
            (d_test / n).write_bytes(ss.test_fotos[n])                       # original
            cv2.imwrite(str(d_res / n), ss.test_cache[(n,) + clave][0])      # con detección
        tabla.to_csv(d_res / "resumen.csv", index=False, encoding="utf-8")
        comparacion.to_csv(d_res / "comparacion.csv", index=False, encoding="utf-8")
        st.success(f"Guardado: fotos en {d_test} y resultados en {d_res}")
