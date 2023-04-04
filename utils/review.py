import streamlit as st

import pandas as pd


def literature_review():

    html_temp = """
        <div>
        <h2></h2>
        <center><h3>Literature Review</h3></center>
        <hr>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")

    df = pd.read_csv("docs/literature_review/review.csv", index_col=False)
    df.reset_index(drop=True, inplace=True)
    print(df)

    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    st.table(df[1:][:])