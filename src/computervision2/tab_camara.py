"""Pestaña 'Cámara en vivo': WebRTC + todos los modelos activos."""
import threading

import av
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from computervision2.inferencia import detectar

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class YOLOVideoProcessor(VideoProcessorBase):
    """Corre en un hilo aparte: recibe cada fotograma, lo anota y lo devuelve."""

    def __init__(self):
        self.modelos = []
        self.conf = 0.25
        self.lock = threading.Lock()  # protege modelos/conf, que se cambian desde el hilo principal

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            modelos, conf = self.modelos, self.conf
        if modelos:
            img, _ = detectar(modelos, img, conf)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render(modelos, conf):
    st.subheader("Detección en tiempo real")
    st.caption("Inicia la transmisión y permite el acceso a la cámara del navegador.")

    ctx = webrtc_streamer(
        key="yolo-camera",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Si cambias modelos o confianza en la barra lateral, el procesador en marcha se actualiza solo
    if ctx.video_processor:
        with ctx.video_processor.lock:
            ctx.video_processor.modelos = modelos
            ctx.video_processor.conf = conf

    st.caption("Con varios modelos los FPS bajan; si va lento, usa menos modelos o uno más ligero.")
