import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

import requests
import base64
from base64 import decodebytes
import io
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


def main():
    st.header("Real-time Streamlit Logo Detection with Roboflow")

    logo_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f" {thread.name} ({thread.ident})")


def logo_detection():
    """Streamlit Logo Detection with Roboflow
    """
    ROBOFLOW_SIZE = 720
    FRAMERATE = 30
    BUFFER = 0.1
    parts = []
    url_base = 'https://detect.roboflow.com/'
    endpoint = 'srwebinar/1'
    access_token = '?api_key=RZZN2hwLn9O50hUmoA6I'
    format = '&format=json'
    confidence = '&confidence=10'
    stroke='&stroke=1'
    parts.append(url_base)
    parts.append(endpoint)
    parts.append(access_token)
    parts.append(format)
    parts.append(confidence)
    parts.append(stroke)
    url = ''.join(parts)
    headers = {'accept': 'application/json'}


    class RoboflowVideoProcessor(VideoProcessorBase):

        def __init__(self) -> None:
            None

        def _annotate_image(self, image, detections):
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()

            for box in detections:
                color = "#4892EA"
                x1 = box['x'] - box['width'] / 2
                x2 = box['x'] + box['width'] / 2
                y1 = box['y'] - box['height'] / 2
                y2 = box['y'] + box['height'] / 2

                draw.rectangle([
                    x1, y1, x2, y2
                ], outline=color, width=3)

                if True:
                    text = box['class']
                    text_size = font.getsize(text)

                    # set button size + 10px margins
                    button_size = (text_size[0]+20, text_size[1]+20)
                    button_img = Image.new('RGBA', button_size, color)
                    # put text on button with 10px margins
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

                    # put button on source image in position (0, 0)
                    image.paste(button_img, (int(x1), int(y1)))
            return np.asarray(image)


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            
            # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
            height, width, channels = image.shape
            scale = ROBOFLOW_SIZE / max(height, width)
            image = cv2.resize(image, (round(scale * width), round(scale * height)))

            # Encode image to base64 string
            retval, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer)
            img_str = img_str.decode("ascii")

            # start = time.time()
            resp = requests.post(url, data=img_str, headers=headers)
            # print('\npost took ' + str(time.time() - start))

            preds = resp.json()
            detections = preds['predictions']

            annotated_image = self._annotate_image(image, detections)

            # self._futures.append(annotated_image)

            # if len(self._futures) < BUFFER * FRAMERATE:
            #     return av.VideoFrame.from_ndarray(image, format="bgr24")
            
            # annotated_image = self._futures.pop(0)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="logo-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=RoboflowVideoProcessor,
        async_processing=True,
    )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()