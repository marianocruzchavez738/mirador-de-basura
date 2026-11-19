"""Función de detección única que usan todas las pestañas (imagen, video, cámara, testeo)."""
import cv2


def detectar(modelos, img_bgr, conf=0.25):
    """Pasa la imagen por todos los modelos y dibuja todas las cajas en una sola imagen.

    Devuelve (imagen_anotada_bgr, detecciones), donde detecciones es una lista de
    diccionarios {"modelo", "clase", "confianza"}.
    """
    lienzo = img_bgr.copy()
    escala = max(lienzo.shape[1] / 1000, 0.5)  # el texto crece con el ancho de la imagen
    grosor = max(1, int(escala * 2))
    detecciones = []

    for m in modelos:
        resultado = m.yolo.predict(img_bgr, conf=conf, verbose=False)[0]
        for c in resultado.boxes:
            x1, y1, x2, y2 = map(int, c.xyxy[0].tolist())
            clase = resultado.names[int(c.cls)]  # nombre de clase según ESTE modelo
            confianza = float(c.conf)
            cv2.rectangle(lienzo, (x1, y1), (x2, y2), m.color, grosor)
            cv2.putText(lienzo, f"{clase} {confianza:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * escala, m.color, grosor)
            detecciones.append({"modelo": m.nombre, "clase": clase, "confianza": round(confianza, 2)})

    # Con varios modelos, leyenda arriba a la izquierda: qué color es cada uno
    if len(modelos) > 1:
        for i, m in enumerate(modelos):
            cv2.putText(lienzo, m.nombre, (10, int(30 * escala) * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * escala, m.color, grosor)

    return lienzo, detecciones


def contar_por_modelo(detecciones, modelos):
    """{'basurav2': 3, 'basurav6': 1} -> cuántas detecciones hizo cada modelo."""
    return {m.nombre: sum(1 for d in detecciones if d["modelo"] == m.nombre) for m in modelos}


def resumen_texto(detecciones):
    """Texto tipo 'basurav6: bolsa (0.87); basurav2: resto (0.40)'."""
    return "; ".join(f"{d['modelo']}: {d['clase']} ({d['confianza']:.2f})" for d in detecciones)
