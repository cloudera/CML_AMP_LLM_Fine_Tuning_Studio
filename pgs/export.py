import streamlit as st


with st.container(border=True):
    col1, col2 = st.columns([1,17])
    with col1:
        col1.image("./resources/images/move_group_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Export to CML Model Registry', divider='red')
        st.caption("Export your fine-tuned models to the Cloudera Model Registry for easy access and deployment.")

st.write("\n")

st.info("This Feature is coming soon !", icon=":material/info:")
