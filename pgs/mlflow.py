import streamlit as st


with st.container(border=True):
    col1, col2 = st.columns([1,17])
    with col1:
        col1.image("./resources/images/model_training_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
    with col2:
        col2.subheader('Run MLFlow Evaluation', divider='red')
        st.caption("Execute comprehensive MLFlow evaluations on your fine-tuned model to ensure accuracy, performance, and reliability, gaining valuable insights.")

st.write("\n")

st.info("This Feature is coming soon !", icon=":material/info:")