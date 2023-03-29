from inference import LinkNetSeg

from utils.home import *
from utils.architecture import *
from utils.visualizer import *

import streamlit as st
from PIL import Image
import time
import urllib.request
import cv2
import numpy as np


opt = st.sidebar.selectbox("Main",("Home", "Architecture", "Visualizer"), label_visibility="hidden")

if opt == "Home":

    home()


elif opt == "Architecture":

    architecture()


elif opt == "Visualizer":

    try:

        input_image = uploader()

        if input_image is not None:

            center(input_image)

            test_path = writer(input_image)

            # Initializing the LinkSeg Inference
            lns = LinkNetSeg(test_path)

            # Running inference
            lns.inference(set_weight_dir = 'linknet.pth', path = 'misc/streamlit_downloads/output.png', blend_path = 'misc/streamlit_downloads/blend.png')

            if st.button("Segment!"):
                display()

    except Exception as e:
        pass