import streamlit as st

from PIL import Image
import cv2
import time
import urllib.request
import numpy as np

def uploader():

    upload_method = st.selectbox("How do you want to upload the image for segmentation?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))

    if upload_method == 'Upload image from device':

        file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is not None:
            image = Image.open(file)

    elif upload_method == 'Upload image via link':

        try:
            img = st.text_input('Enter the Image Address')
            image = Image.open(urllib.request.urlopen(img))
            
        except:
            if st.button('Submit'):
                show = st.error("Please Enter a valid Image Address!")
                time.sleep(4)
                show.empty()

    return image

def writer(image):

    image = np.array(image)
    test_image = image[:, :, ::-1].copy()

    test_path = "misc/streamlit_uploads/input.png"
    cv2.imwrite(test_path, test_image)

    return test_path

def columns(imglist, captions):

    try:
        idx = 0

        while idx < len(imglist):
            
            for _ in range(len(imglist)):
                cols = st.columns(2) 

                for col_num in range(2): 

                    if idx <= len(imglist):
                        cols[col_num].image(imglist[idx], 
                            width=328, caption=captions[idx])
                        
                        idx+=1
                        
    except:

        pass

def center(image):

    col1, col2, col3 = st.columns([4,10,4])
    
    with col1:
        st.write("")
    with col2:
        st.image(image, width=325, caption = "Uploaded Image")
    with col3:
        st.write("")

    st.title("")

def display():

    mask_image = Image.open("misc/streamlit_downloads/output.png")
    blend_image = Image.open("misc/streamlit_downloads/blend.png")

    images = [mask_image, blend_image]
    captions = ["Mask Image", "Blend Image"]

    columns(images, captions)