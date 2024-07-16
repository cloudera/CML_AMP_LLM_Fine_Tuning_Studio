import streamlit as st


st.title("Data Page")

st.header("Upload Data")
data_file = st.file_uploader("Upload a dataset", type=["csv", "txt", "json"])

if data_file is not None:
    st.write("Data file uploaded successfully")
    # Add data handling logic here
