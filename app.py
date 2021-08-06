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

### setting up the logger to log messages created by the script to help debug in main()
logger = logging.getLogger(__name__)

#### For the webrtc plug in, this is setting the default values for the widget,
#### incluing removing the audio
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


def main():
    """
    Main part of the script that defines the main method and calls the logo detection.
    Also sets up a logger for debugging purposes.
    """
    st.header("Real-time Streamlit Logo Detection with Roboflow")

    logo_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f" {thread.name} ({thread.ident})")


def logo_detection():
    """Streamlit Logo Detection with Roboflow"""

    ##########
    ##### Set up sidebar
    ##########

    st.sidebar.write("### Streamlit Logo Detection")

    ## Add in sliders.
    CONFIDENCE_THRESHOLD = st.sidebar.slider(
        "Confidence threshold:",
        0,
        100,
        50,
        5,
        help="What is the minimum acceptable confidence level for displaying a bounding box?",
    )
    OVERLAP_THRESHOLD = st.sidebar.slider(
        "Overlap threshold:",
        0,
        100,
        30,
        5,
        help="What is the maximum amount of overlap permitted between visible bounding boxes?",
    )

    ### Adding in the Streamlit and Roboflow logos to the sidebar
    image = Image.open("./images/roboflow_logo.png")
    st.sidebar.image(image, use_column_width=True)

    image = Image.open("./images/streamlit_logo.png")
    st.sidebar.image(image, use_column_width=True)

    ### Setting up the url and query parameters for roboflow endpoint
    # the overlap and confidence query parameters are set within the
    # RoboflowVideoProcessor class, normally they would be hardcoded here but
    # to get interactivity we had to add them after
    ROBOFLOW_SIZE = 720
    url_base = "https://detect.roboflow.com/"
    endpoint = "merged_logos/2"
    ## remove random part and add your secret key here
    ## Create a .streamlit/secrets.toml with the entry, replacing YourKey with the key from Roboflow: api_key="YourKey"
    ## Don't commit secrets.toml. On Sharing, add the same line to ☰ -> Settings -> Secrets
    access_token = f'?api_key={st.secrets["api_key"]}'
    format = "&format=json"
    headers = {"accept": "application/json"}

    # Map detected classes to uniquely colored bounding boxes
    color_map = {
        "dark logo": "#D41159",
        "old logo": "#1A85FF",
        "white logo": "#FFC20A",
        "roboflow_logomark": "#1AFF1A",
        "roboflow_wordmark": "#582832",
    }

    class RoboflowVideoProcessor(VideoProcessorBase):
        _overlap = OVERLAP_THRESHOLD
        _confidence = CONFIDENCE_THRESHOLD

        def __init__(self) -> None:
            """
            Initalizes the value of the overlap and confidence thresholds based on user input,
            from the sliders in the sidebar.
            """
            self._overlap = OVERLAP_THRESHOLD
            self._confidence = CONFIDENCE_THRESHOLD

        def set_overlap_confidence(self, overlap, confidence):
            """
            updating the values of overlap and condidence based on slider values
            once the video is running.
            """
            self._overlap = overlap
            self._confidence = confidence

        # Draw bounding boxes from the inference API JSON output
        def _annotate_image(self, image, detections):
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()

            ### This is copied from the roboflow docs line (143-166):
            ### https://docs.roboflow.com/inference/hosted-api#drawing-a-box-from-the-inference-api-json-output
            ### line 143 was added to map the detected classes to unique colors
            ### detection is list = [{'class', x,y,width,height}, repeating]
            for box in detections:
                color = color_map[box["class"]]
                x1 = box["x"] - box["width"] / 2
                x2 = box["x"] + box["width"] / 2
                y1 = box["y"] - box["height"] / 2
                y2 = box["y"] + box["height"] / 2

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                if True:
                    text = box["class"]
                    text_size = font.getsize(text)

                    # set button size + 10px margins
                    button_size = (text_size[0] + 20, text_size[1] + 20)
                    button_img = Image.new("RGBA", button_size, color)
                    # put text on button with 10px margins
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text(
                        (10, 10), text, font=font, fill=(255, 255, 255, 255)
                    )

                    # put button on source image in position (0, 0)
                    image.paste(button_img, (int(x1), int(y1)))
            return np.asarray(image)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """
            Recieves images and sends annotated images:
            gets webcam image, does some preprocessing and calls the
            annotated function above to put bounding boxes on the detections and
            finally returns the annotated video frame.
            """
            image = frame.to_ndarray(format="bgr24")

            # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
            # roboflow_size is set in logo_detection when we set up the url and query parameters
            # hardcoded to 720 currently
            height, width, channels = image.shape
            scale = ROBOFLOW_SIZE / max(height, width)
            image = cv2.resize(image, (round(scale * width), round(scale * height)))

            # Encode image to base64 string formats becasue that is what the
            # endpoint expects
            retval, buffer = cv2.imencode(".jpg", image)
            img_str = base64.b64encode(buffer)
            img_str = img_str.decode("ascii")

            ### construct the request url with the confidence and overlap
            parts = []
            overlap = f"&overlap={self._overlap}"
            confidence = f"&confidence={self._confidence}"
            parts.append(url_base)
            parts.append(endpoint)
            parts.append(access_token)
            parts.append(format)
            parts.append(overlap)
            parts.append(confidence)
            url = "".join(parts)

            # making the post request to the roboflow endpoint with the url we just made
            resp = requests.post(url, data=img_str, headers=headers)

            ## from the responce, we convert it to a JSON and get the predictions
            preds = resp.json()
            detections = preds["predictions"]

            ### finally, annotate the image
            annotated_image = self._annotate_image(image, detections)

            ## return the image with the bounding boxes to the browser
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    ### reach out to compnent developer about 221-227
    ## likely, just setting up the component itself, the mode to both send and recieve data
    # pass the WEBRTC_CLIENT_SETTINGS we choose and tell it to process the video
    # frames using the logic from the RoboflowVideoProcessor class
    webrtc_ctx = webrtc_streamer(
        key="logo-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=RoboflowVideoProcessor,
        async_processing=True,
    )

    ## calling the overlap function with the actual values returned from the sliders
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_overlap_confidence(
            OVERLAP_THRESHOLD, CONFIDENCE_THRESHOLD
        )


### the main program where we call the main method
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
