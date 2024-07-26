import streamlit as st

def load_markdown_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def create_homepage_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 16])
        with col1:
            col1.image("./resources/images/architecture_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png")
        with col2:
            col2.subheader('LLM Finetuning Studio', divider='red')
            col2.caption(
                'The LLM Fine Tuning Studio, updated in July 2024, features a new Streamlit-based UI and integrates with Cloudera Machine Learning (CML) components. '
                'It supports custom datasets, BitsAndBytes, LoRA configurations, and distributed training.'
            )

def create_tile(container, image_path, button_text, page_path, description):
    tile = container.container(height=160)
    if tile.button(button_text, type="primary", use_container_width=True):
        st.switch_page(page_path)
    c1, c2 = tile.columns([1, 4])
    with c1:
        c1.image(image_path)
    with c2:
        c2.caption(description)

def create_tab(tab_name, tiles):
    with tab_name:
        col1, col2, col3, col4 = st.columns(4)
        if tiles[0]:
            tiles[0](col1)
        if tiles[1]:
            tiles[1](col2)
        if tiles[2]:
            tiles[2](col3)
        if tiles[3]:
            tiles[3](col4)


create_homepage_header()
st.write("\n")  

tab1, tab2, tab3, tab4 = st.tabs(["**Datasets**", "**Models & Prompts**", "**Experiments**", "**Model Management**"])

create_tab(tab1, [
    lambda container: create_tile(container, "./resources/images/publish_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png", 
                                    "Import Datasets", "pgs/datasets.py", 
                                    'Import datasets from Hugging Face or upload your own preprocessed dataset from local sources for fine-tuning.'),
    lambda container: create_tile(container, "./resources/images/data_object_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png", 
                                    "View Datasets", "pgs/view_datasets.py", 
                                    'Explore the imported datasets and gain insights into the structure and content of each dataset.'),
    lambda container: None,  # Add a third tile here if needed
    lambda container: None  # Add a third tile here if needed
])

create_tab(tab2, [
    lambda container: create_tile(container, "./resources/images/neurology_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png", 
                                    "Import Models & Adapters", "pgs/models.py", 
                                    'Import foundational LLM models from Hugging Face or local sources for your fine-tuning job specific requirements.'),
    lambda container: create_tile(container, "./resources/images/chat_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png", 
                                    "Create Prompts", "pgs/prompts.py", 
                                    'Generate tailored prompts for your fine-tuning tasks on the specified datasets and models to enhance performance.'),
    lambda container: None,  # Add a third tile here if needed
    lambda container: None  # Add a third tile here if needed
])

create_tab(tab3, [
    lambda container: create_tile(container, "./resources/images/forward_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png", 
                                    "Finetune your model", "pgs/train_adapter.py", 
                                    'Finetune your model, leveraging advanced techniques to improve performance.'),
    lambda container: create_tile(container, "./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png", 
                                    "Monitor Finetuning Jobs", "pgs/jobs.py", 
                                    'Monitor your fine-tuning jobs and track progress throughout the training process.'),
    lambda container: create_tile(container, "./resources/images/difference_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png", 
                                    "Local Adapater Comparison", "pgs/evaluate.py", 
                                    'Compare your fine-tuned model performance with base model and gain valuable insights.'),
    lambda container: create_tile(container, "./resources/images/model_training_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png", 
                                    "Run MLFlow Evaluation", "pgs/mlflow.py", 
                                    'Execute MLFlow evaluations on your fine-tuned model to ensure accuracy, performance, gaining valuable insights.')
])

create_tab(tab4, [
    lambda container: create_tile(container, "./resources/images/move_group_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png", 
                                    "Export to CML Model Registry", "pgs/export.py", 
                                    'Export your fine-tuned models to the Cloudera Model Registry for easy access and deployment.'),
    lambda container: create_tile(container, "./resources/images/deployed_code_24dp_EA3323_FILL0_wght400_GRAD0_opsz48.png", 
                                    "Deploy to Cloudera AI Inference", "pgs/deploy.py", 
                                    'Deploy your models to Cloudera AI Inference for production use and real-world application deployment.'),
    lambda container: None,  # Add a third tile here if needed
    lambda container: None  # Add a third tile here if needed
])