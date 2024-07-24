import streamlit as st

# Uncomment the following line if you want to include a markdown file at the top of your app
# st.markdown(open("pgs/home.md").read())

with st.container(border=True):
    col1, col2 = st.columns([1,8])
    with col1:
        col1.image("./resources/images/diapason-tuner-svgrepo-com.svg", use_column_width=True)
    with col2:
        col2.subheader('LLM Finetuning Studio', divider='rainbow')
        col2.write('The LLM Fine Tuning Studio, updated in July 2024, features a new Streamlit-based UI and integrates with Cloudera Machine Learning (CML) components. It supports custom datasets, BitsAndBytes, LoRA configurations, and distributed training.')

st.write("\n")  

# Initialize session state if not already done
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Datasets'

# Function to change the selected tab
def select_tab(tab_name):
    st.session_state.selected_tab = tab_name

tab1, tab2, tab3, tab4 = st.tabs(["Datasets", "Models & Prompts", "Experiments", "Model Management"])

with tab1:
    tab1Col1, tab1Col2, tab1Col3 = st.columns(3)

    with tab1Col1:
        tile = st.container(height=200)
        if tile.button("Import Datasets", type="primary", use_container_width=True):
            st.switch_page("pgs/datasets.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/file-import-arrow-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datasets from either available datsets on huggingface or upload your own preprocessed dataset from local.')

    with tab1Col2:
        tile = st.container(height=200)
        if tile.button("View Datasets", type="primary", use_container_width=True):
            st.switch_page("pgs/view_datasets.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/grid-view-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Explore and organize imported datasets from Hugging Face or custom sources. Gain insights into the structure and content of each dataset.')
            


with tab2:
    tab2Col1, tab2Col2, tab2Col3 = st.columns(3)

    with tab2Col1:
        tile = st.container(height=200)
        if tile.button("Import Models & Adapaters", type="primary", use_container_width=True):
            st.switch_page("pgs/models.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/data-exploration-data-center-model-management-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import foundational LLM models from Hugging Face or local sources to align with your finetuning job specific requirements.')
        
    with tab2Col2:
        tile = st.container(height=200)
        if tile.button("Create Prompts", type="primary", use_container_width=True):
            st.switch_page("pgs/prompts.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/system-edit-line-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Generate tailored prompts for your fine-tuning tasks on the specified datasets and models.')

with tab3:
    tab3Col1, tab3Col2, tab3Col3 = st.columns(3)

    with tab3Col1:
        tile = st.container(height=200)
        if tile.button("Finetune your model", type="primary", use_container_width=True):
            st.switch_page("pgs/train_adapter.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/diapason-tuner-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datsets from either available datsets on huggingface or upload your own preprocessed dataset from local.')

    with tab3Col2:
        tile = st.container(height=200)
        if tile.button("Monitor Finetuning Jobs", type="primary", use_container_width=True):
            st.switch_page("pgs/jobs.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/tree-view-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datsets from either available datsets on huggingface or upload your own preprocessed dataset from local.')
    
    with tab3Col3:
        tile = st.container(height=200)
        if tile.button("View Evaluation Results", type="primary", use_container_width=True):
            st.switch_page("pgs/evaluate.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/scale-balanced-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datsets from either available datsets on huggingface or upload your own preprocessed dataset from local.')

with tab4:
    tab4Col1, tab4Col2, tab4Col3 = st.columns(3)

    with tab4Col1:
        tile = st.container(height=200)
        if tile.button("Export to CML Model Registry", type="primary", use_container_width=True):
            st.switch_page("pgs/export.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/object-connection-1087-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datsets from either available datsets on huggingface or upload your own preprocessed dataset from local.')
        
    with tab4Col2:
        tile = st.container(height=200)
        if tile.button("Deploy to Cloudera AI Inference", type="primary", use_container_width=True):
            st.switch_page("pgs/deploy.py")
        tile.write("\n")
        c1, c2 = tile.columns([1,4])
        with c1:
            c1.image("./resources/images/model-alt-svgrepo-com.svg", use_column_width=True)
        with c2:
            c2.caption('Import the datsets from either available datsets on huggingface or upload your own preprocessed dataset from local.')