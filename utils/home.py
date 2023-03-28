import streamlit as st

def home():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>LinkSeg</h3></center>
        </div>
        <hr>
        <div style = "text-align:justify">
        The characteristics of Retinal Blood Vessels help diagnose various eye ailments. The proper localization, extraction and segmentation of the blood vessels is essential to the cause of treatment of the eye. Manual segmentation of blood vessels may be error prone and inaccurate leading to difficulty in further treatment. We present a novel approach of semantic segmentation of Retinal Blood Vessels using Linked Networks to account for lost spatial information during feature extraction. The implementation of the segmentation technique involves using Residual Networks as a feature extractor and Transpose Convolution for image to image translation thereby giving a segmentation mask as an output. The main feature of the architecture is the links between the Feature Extractor and the Decoder networks that enhance the performance of the network by helping in the recovery of lost spatial information. Training and Validation using the Pytorch framework has been performed on the Digital Retinal Images for Vessel Extraction (DRIVE) Dataset to establish quality results.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    prob = st.checkbox("Problem")

    if prob:
        st.write("Manual segmentation of Retinal Blood Vessels is often error prone and exhausting for skilled ophthalmologists, leading to improper diagnosis of the eye ailments.")

    obj = st.checkbox("Objective")

    if obj:
        st.write("Implementations of a Semantic Segmentation Network to accurately segment the Blood Vessels present in the retina for proper treatment and reduction of operator fatigue.")

    res = st.checkbox("Check out the Results of our Implementation!")

    if res:
        st.image("misc/results/results.png", caption="Results")