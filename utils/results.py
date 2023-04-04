import streamlit as st
import streamlit.components.v1 as components

import pandas as pd

def results():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>Results and Analysis</h3></center>
        <hr>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    st.write("Validation was performed on the DRIVE Dataset and the STARE Dataset using NVIDIA Tesla T4 GPU. The weights were hosted and downloaded via gdown for further tests. The Figure on the next slide shows the test results of the input retinal images from DRIVE Dataset and validation results are mentioned in the Table below.")

    scores = pd.DataFrame(
        {
        "Metrics": ["Dice Loss", "IoU Loss"], 
        "DRIVE": ["0.1164", "0.2086"],
        "STARE": ["0.1277", "0.1962"]
        }
    )
    scores.reset_index(drop=True)

    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    st.table(scores)

    st.write("A comparison of our architecture with state-of-the-art architecturs has been shown in the Table below")

    sota = pd.DataFrame(
        {
        "Architecture": [
            "UNETs", 
            "Lattice NN with Dendrite Processing",
            "Multi-level CNN with Conditional Random Fields", 
            "Multi-scale Line Detection", 
            "CLAHE", 
            "Modified SUSAN edge detector", 
            "Improved LinkNet"
            ], 
        "DRIVE": [
            "0.9790 (AUC ROC)", 
            "0.81 (F1 Score)", 
            "0.9523 (Accuracy)", 
            "0.9326 (Accuracy)", 
            "0.9477 (Accuracy)", 
            "0.9633 (Accuracy)", 
            "0.8836 (Dice Score)"
            ]
        }
    )
    sota.reset_index(drop=True)

    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    st.table(sota)
    st.write("")

    imageCarouselComponent = components.declare_component("image-carousel-component", path="utils/frontend/public")

    imageUrls = [
                "https://user-images.githubusercontent.com/66861243/229856771-af1b3144-55d2-473a-acf8-35c548436192.png",
                "https://user-images.githubusercontent.com/66861243/229853783-558ed22d-59f4-45c7-9686-2464fb257a3f.png",
                "https://user-images.githubusercontent.com/66861243/229851588-e512a256-541b-4930-a98c-818f790b9eb1.png",
                "https://user-images.githubusercontent.com/66861243/229852960-43d27837-9df5-47eb-8d0a-43616de8a2c7.png",
                "https://user-images.githubusercontent.com/66861243/229852972-da026db6-8405-4c7a-ab14-92fa6106dce7.png",
                "https://user-images.githubusercontent.com/66861243/229849502-7ceeb590-7aaa-4264-902c-0037dcf20dbc.png",
        ]
    
    st.write("Results Gallery")

    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)