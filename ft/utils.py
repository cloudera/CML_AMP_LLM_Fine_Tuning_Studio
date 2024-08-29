import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import os
import requests
import torch
from huggingface_hub import login
import yaml
from ft.consts import AXOLOTL_TRAINING_CONFIGS_TEMPLATE_FILE_PATH


def get_env_variable(var_name: str, default_value: Optional[str] = None) -> str:
    """Get environment variable or return default value."""
    value = os.getenv(var_name)
    if not value:
        if default_value:
            return default_value
        st.error(f"Environment variable '{var_name}' is not set.")
        st.stop()
    return value


def fetch_resource_usage_data(host: str, project_owner: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch data from API and handle errors."""
    url = f"{host}/users/{project_owner}/resources-usage"
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(api_key, ""))
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return None
    
def fetch_cml_site_config(host: str, project_owner: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch data from API and handle errors."""
    url = f"{host}/site/config/"
    try:
        res = requests.get(url, headers={"Content-Type": "application/json"}, auth=(api_key, ""))
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch siteconfig: {e}")
        return None


def load_markdown_file(file_path: str) -> str:
    """Load and return the content of a markdown file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return ""


def get_device() -> torch.device:
    """
    Get the type of device used to load models and tensors during this session.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def attempt_hf_login(access_token):
    # Try to log in to HF hub to access gated models for fine tuning.
    try:
        if access_token is not None:
            login(access_token)
    except Exception as e:
        print("Could not log in to HF! Cannot use gated models.")
        print(e)


def process_resource_usage_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Process the JSON data to extract relevant information and return a DataFrame."""
    cluster_data = data.get('cluster', {})

    resources = [
        ('cpu', 'cpuAllocatable', 'vCPU', None),
        ('memory', 'memoryAllocatable', 'GiB', None),  # Convert from bytes to GiB
        ('nvidiaGPU', 'nvidiaGPUAllocatable', 'GPU', None)
    ]
    rows = []

    for resource, allocatable_field, unit, conversion_factor in resources:
        current_usage = cluster_data.get(resource, 0)
        max_usage = cluster_data.get(allocatable_field, 0)

        if conversion_factor:
            max_usage_value = float(max_usage) / conversion_factor
        else:
            max_usage_value = float(max_usage)

        rows.append({
            'Resource Name': resource.upper(),
            'Progress': current_usage / max_usage_value * 100 if max_usage_value else 0,
            'Used': f"{current_usage:.2f} {unit}",
            'Max Available': f"{max_usage_value-current_usage:.2f} {unit}"
        })

    return pd.DataFrame(rows)


def dict_to_yaml_string(yaml_dict):
    """
    Convert a Python dictionary to a YAML string, with custom handling for None values.
    None values will be represented as empty strings in the YAML output.

    :param yaml_dict: The dictionary to be converted to a YAML string.
    :return: A string containing the YAML representation of the dictionary.
    """
    if not isinstance(yaml_dict, dict):
        raise TypeError("Input must be a dictionary")

    # Custom representer to handle empty strings for None values
    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

    # Register the custom representer for NoneType
    yaml.add_representer(type(None), represent_none)

    try:
        # Convert the dictionary to a YAML string
        yaml_string = yaml.dump(yaml_dict, default_flow_style=False)
    except yaml.YAMLError as e:
        raise ValueError(f"Error converting dictionary to YAML string: {e}")

    return yaml_string


def save_yaml_file(yaml_dict, file_path):
    """
    Save a Python dictionary as a YAML file, with custom handling for None values.
    None values will be represented as empty strings in the YAML output.

    :param yaml_dict: The dictionary to be saved as a YAML file.
    :param file_path: The path where the YAML file should be saved.
    """
    if not isinstance(yaml_dict, dict):
        raise TypeError("Input must be a dictionary")

    # Custom representer to handle empty strings for None values
    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

    # Register the custom representer for NoneType
    yaml.add_representer(type(None), represent_none)

    try:
        # Save the dictionary as a YAML file
        with open(file_path, 'w') as file:
            yaml.dump(yaml_dict, file, default_flow_style=False)
    except (IOError, yaml.YAMLError) as e:
        raise IOError(f"Error saving YAML file to {file_path}: {e}")

# Modify this function to handle unexpected types safely


def format_status_with_icon(status):
    if not isinstance(status, str):
        status = 'Unknown'  # Default to 'Unknown' if status is not a string
    icons = {
        "succeeded": "🟢 Succeeded",
        "running": "🔵 Running",
        "scheduling": "🟡 Scheduling",
        "failed": "🔴 Failed",
        "Unknown": "⚪ Unknown"
    }
    return icons.get(status, f"⚪ {status.capitalize()}")


def get_axolotl_training_config_template_yaml_str():
    with open(AXOLOTL_TRAINING_CONFIGS_TEMPLATE_FILE_PATH, 'r') as file:
        yaml_content = file.read()
    return yaml_content

def generate_templates(columns):
    output_column_names = [
        "label", "response", "answer", "target", "output", "sentiment", "toxicity", "rating", 
        "classification", "label_text", "summary", "translation", "emotion", "verdict", "decision", 
        "output_text", "stance", "score", "rank", "opinion", "relevance", "truth", "gold_label", 
        "sentiment_score", "category", "label_id", "target_text", "intent", "pred", "prediction", 
        "logits", "is_hate_speech", "is_spam", "score1", "output1", "human_label", "is_humor", "fact", 
        "is_sarcasm", "category_label", "true_label", "response_text", "alignment_score", 
        "classification_label", "stance_label", "verdict_label", "truth_label", "final_score", 
        "answer_text", "diagnosis", "rating_label", "output_label", "is_positive", "summary_text", 
        "hate_speech_label", "sarcasm_label", "opinion_label", "emotion_label", "quality_label", 
        "alignment_label", "category_text", "stance_score", "humor_label", "is_offensive", "spam_label", 
        "fact_score", "final_label", "decision_label", "gold_standard", "prediction_text", 
        "emotion_score", "intent_label", "opinion_text", "stance_text", "humor_score", "is_fake_news", 
        "is_true", "sarcasm_score", "spam_score", "final_decision", "output_final", "is_correct", 
        "label_final", "is_positive_sentiment", "output_summary", "toxicity_score", "rating_score", 
        "truth_score", "is_toxic", "prediction_final", "gold_label_text", "category_final", 
        "output_gold", "final_opinion", "stance_final", "sarcasm_final", "spam_final", "humor_final", 
        "output_humor", "final_verdict"
    ]

    # Identify input and output columns based on predefined output column names
    output_columns = [col for col in columns if col.lower() in output_column_names]
    input_columns = [col for col in columns if col.lower() not in output_column_names]

    # If no output columns found, split the dataset columns into input and output halves
    if not output_columns:
        half_idx = len(columns) // 2
        input_columns = columns[:half_idx]
        output_columns = columns[half_idx:]

    # Generate default prompt template
    prompt_template = "You are an LLM responsible with generating a response. Please provide a response given the user input below.\n\n"
    for feature in input_columns:
        prompt_template += f"<{feature.capitalize()}>: {{{feature}}}\n"

    # Add the first output column placeholder to the prompt template
    if output_columns:
        prompt_template += f"<{output_columns[0].capitalize()}>: \n"

    # Generate default completion template
    completion_template = ""
    if output_columns:
        # Include the first output column directly
        completion_template += f"{{{output_columns[0]}}}\n"
        # Add the remaining output columns with their respective labels
        for feature in output_columns[1:]:
            completion_template += f"<{feature.capitalize()}>: {{{feature}}}\n"

    return prompt_template, completion_template
