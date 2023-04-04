import streamlit as st

def workflow():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>Workflow</h3></center>
        <hr>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    st.info("The complete workflow of the model training has been shown in the diagram below!")
    st.header("")

    st.image("docs/workflow/workflow.png", caption="Workflow")

    st.markdown(
        """
        - The input image of size (512x512) is first passed through augmentations (horizontal flip, vertical flip and rotation).
        - The augmented images undergo normalization and standardization so that they are in a Standard Gaussian Distribution (zero mean and unit standard deviation).
        - The normalized images are the passed to the network for training that generates the output segmentation mask.
        """)
    st.header("")

    st.image("docs/workflow/sequence.png", caption="Sequence Diagram")