import streamlit as st

def home():

    html_temp = """
        <div>
        <h2></h2>
        <center><h2>LinkSeg</h2></center>
        </div>
        <hr>
        <div style = "text-align:justify">
        The characteristics of Retinal Blood Vessels help diagnose various eye ailments. The proper localization, extraction and segmentation of the blood vessels is essential to the cause of treatment of the eye. Manual segmentation of blood vessels may be error prone and inaccurate leading to difficulty in further treatment. We present a novel approach of semantic segmentation of Retinal Blood Vessels using Linked Networks to account for lost spatial information during feature extraction. The implementation of the segmentation technique involves using Residual Networks as a feature extractor and Transpose Convolution for image to image translation thereby giving a segmentation mask as an output. The main feature of the architecture is the links between the Feature Extractor and the Decoder networks that enhance the performance of the network by helping in the recovery of lost spatial information. Training and Validation using the Pytorch framework has been performed on the Digital Retinal Images for Vessel Extraction (DRIVE) Dataset to establish quality results.
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    st.image("docs/dataloader/annot.png", caption="DRIVE Dataset Annotations") 


    prob = st.checkbox("Problem")

    if prob:
        st.write("Manual segmentation of Retinal Blood Vessels is often error prone and exhausting for skilled ophthalmologists, leading to improper diagnosis of the eye ailments.")

    obj = st.checkbox("Objective")

    if obj:
        st.write("Implementations of a Semantic Segmentation Network to accurately segment the Blood Vessels present in the retina for proper treatment and reduction of operator fatigue.")

    limit = st.checkbox("Research Gap")

    if limit:
        st.write("Existing systems for Retinal Blood Vessel Segmentation such as UNETs, CNNs, SVMs have a common problem of the loss of important spatial information during the feature extraction stage as a result of multiple downsampling. Hence, the existing models are not able to achieve the accuracy that it could have if the spatial information has been retained.Certain networks such as UNETs use Transpose Convolutions for the Upsampling task. The usage of Transpose Convolution often leads to noisy outputs.")

    proposal = st.checkbox("Proposal")

    if proposal:
        st.markdown(
            """
            In order to counter the problem of lost spatial information, we propose using an improved LinkNet based architecture for semantic segmentation of Retinal Blood Vessels.

            - The network makes use of skip connections to link each Encoder Block to its corresponding Decoder Block, passing on the lost spatial information to be concatenated during Upsampling.

            - Using Upsample instead of Transpose Conv counters the problem of noisy output.
            """)
    
    req = st.checkbox("Requirements")

    if req:
        st.markdown(
            """
            Software Requirements:
            - Google Colab for Training
            - Visual Studio Code
            - Linux Terminal for local Testing
            - Python
            - PyTorch

            Hardware Requirements:
            - NVIDIA Tesla T4 GPU
            - cuDNN VERSION 8005
            """)

    res = st.checkbox("Check out the Results of our Implementation!")

    if res:
        st.image("misc/results/results.png", caption="Results")