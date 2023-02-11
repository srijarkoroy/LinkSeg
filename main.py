import streamlit as st

opt = st.sidebar.selectbox("",("Home", "Architecture", "Visualizer"))

if opt == "Home":

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>LinkSeg</h3></center>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)