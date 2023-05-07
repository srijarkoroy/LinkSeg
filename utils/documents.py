import streamlit as st

import base64

def PDF(file):
 
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf, unsafe_allow_html=True)

def documents():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>Documents</h3></center>
        <hr>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    st.write("Project Report")
    PDF("docs/embeds/report.pdf")
    st.header("")

    st.write("Research Paper")
    PDF("docs/embeds/paper.pdf")
    st.header("")

    st.write("Slides")
    PDF("docs/embeds/slides.pdf")