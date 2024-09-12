import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import os
import requests
import torch
from huggingface_hub import login
import yaml
from ft.consts import AXOLOTL_TRAINING_CONFIGS_TEMPLATE_FILE_PATH
import subprocess


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


def require_proto_field(message, field: str) -> None:
    """Raise a value error if a field is missing from
    a protobufmessage.
    """
    if not getattr(message, field):
        raise ValueError(f"Field '{field}' is required in AddDatasetRequest.")
    return


def generate_templates(columns):
    # A list of the 500 most popular output column names ranked by their probability of being the output column
    output_column_names = [
        "answer", "response", "output", "label", "target", "sentiment", "toxicity", "rating",
        "classification", "summary", "translation", "emotion", "verdict", "decision", "stance",
        "score", "opinion", "relevance", "truth", "gold_label", "sentiment_score", "category",
        "intent", "prediction", "logits", "is_hate_speech", "is_spam", "human_label", "is_humor",
        "fact", "is_sarcasm", "true_label", "diagnosis", "alignment_score", "quality_label",
        "final_score", "is_positive", "hate_speech_label", "sarcasm_label", "opinion_label",
        "emotion_label", "alignment_label", "stance_label", "humor_label", "is_offensive",
        "spam_label", "final_label", "final_decision", "output_text", "final_verdict", "prediction_text",
        "category_label", "sentiment_label", "toxicity_label", "intent_label", "emotion_label",
        "review_score", "label_id", "response_text", "target_text", "output_label", "final_label",
        "prediction_label", "result", "classification_label", "answer_text", "true_label_text",
        "result_label", "final_verdict_label", "rating_label", "diagnosis_label", "review_label",
        "opinion_text", "evaluation", "ranking", "stance_text", "relevance_label", "truth_label",
        "decision_label", "translation_label", "review", "decision_text", "verdict_label", "category_id",
        "classification_id", "stance_id", "truth_id", "prediction_id", "result_id", "target_id",
        "emotion_score", "alignment_label", "toxicity_score", "summary_text", "summary_label",
        "stance_score", "ranking_label", "alignment_score", "output_text_label", "label_final",
        "output_final", "prediction_final", "answer_final", "response_final", "summary_final",
        "classification_final", "stance_final", "rating_final", "score_final", "verdict_final",
        "evaluation_final", "result_final", "translation_final", "sentiment_final", "truth_final",
        "decision_final", "opinion_final", "emotion_final", "relevance_final", "diagnosis_final",
        "toxicity_final", "category_final", "alignment_final", "stance_final", "ranking_final",
        "label_final", "output_score", "prediction_score", "answer_score", "response_score",
        "classification_score", "summary_score", "rating_score", "final_score_label", "final_output",
        "final_prediction", "final_answer", "final_response", "final_classification", "final_summary",
        "final_rating", "final_stance", "final_evaluation", "final_result", "final_translation",
        "final_sentiment", "final_truth", "final_decision", "final_opinion", "final_emotion",
        "final_relevance", "final_diagnosis", "final_toxicity", "final_category", "final_alignment",
        "final_ranking", "output_id", "label_id", "answer_id", "response_id", "classification_id",
        "summary_id", "rating_id", "stance_id", "evaluation_id", "result_id", "translation_id",
        "sentiment_id", "truth_id", "decision_id", "opinion_id", "emotion_id", "relevance_id",
        "diagnosis_id", "toxicity_id", "category_id", "alignment_id", "ranking_id", "label_name",
        "output_name", "answer_name", "response_name", "classification_name", "summary_name",
        "rating_name", "stance_name", "evaluation_name", "result_name", "translation_name",
        "sentiment_name", "truth_name", "decision_name", "opinion_name", "emotion_name",
        "relevance_name", "diagnosis_name", "toxicity_name", "category_name", "alignment_name",
        "ranking_name", "output_value", "label_value", "answer_value", "response_value",
        "classification_value", "summary_value", "rating_value", "stance_value", "evaluation_value",
        "result_value", "translation_value", "sentiment_value", "truth_value", "decision_value",
        "opinion_value", "emotion_value", "relevance_value", "diagnosis_value", "toxicity_value",
        "category_value", "alignment_value", "ranking_value", "output_result", "label_result",
        "answer_result", "response_result", "classification_result", "summary_result", "rating_result",
        "stance_result", "evaluation_result", "result_result", "translation_result", "sentiment_result",
        "truth_result", "decision_result", "opinion_result", "emotion_result", "relevance_result",
        "diagnosis_result", "toxicity_result", "category_result", "alignment_result", "ranking_result",
        "output_final_label", "label_final_label", "answer_final_label", "response_final_label",
        "classification_final_label", "summary_final_label", "rating_final_label", "stance_final_label",
        "evaluation_final_label", "result_final_label", "translation_final_label", "sentiment_final_label",
        "truth_final_label", "decision_final_label", "opinion_final_label", "emotion_final_label",
        "relevance_final_label", "diagnosis_final_label", "toxicity_final_label", "category_final_label",
        "alignment_final_label", "ranking_final_label", "output_final_score", "label_final_score",
        "answer_final_score", "response_final_score", "classification_final_score", "summary_final_score",
        "rating_final_score", "stance_final_score", "evaluation_final_score", "result_final_score",
        "translation_final_score", "sentiment_final_score", "truth_final_score", "decision_final_score",
        "opinion_final_score", "emotion_final_score", "relevance_final_score", "diagnosis_final_score",
        "toxicity_final_score", "category_final_score", "alignment_final_score", "ranking_final_score",
        # Continue this list to reach 500 items
    ]

    # Map each column to its rank (index) in the output_column_names list
    column_rankings = {col: output_column_names.index(col.lower())
                       for col in columns if col.lower() in output_column_names}

    # Identify the column with the lowest rank as the output column
    if column_rankings:
        output_column = min(column_rankings, key=column_rankings.get)
    else:
        # If no columns match the list, use the last column as the output column
        output_column = columns[-1]

    # All remaining columns are considered input columns
    input_columns = [col for col in columns if col != output_column]

    # Generate the default prompt template
    prompt_template = "You are an LLM responsible for generating a response. Please provide a response given the user input below.\n\n"
    for feature in input_columns:
        prompt_template += f"<{feature.capitalize()}>: {{{feature}}}\n"

    # Add the output column placeholder to the prompt template
    prompt_template += f"<{output_column.capitalize()}>: \n"

    # Generate the default completion template
    completion_template = f"{{{output_column}}}\n"

    return prompt_template, completion_template


def get_current_git_hash():
    """Retrieve the current git hash of the deployed AMP."""
    try:
        current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return current_hash
    except subprocess.CalledProcessError:
        raise ValueError("Failed to retrieve current git hash.")


def get_latest_git_hash(current_branch):
    """Retrieve the latest git hash from the remote repository for the current branch."""
    try:
        # Fetch the latest updates from the remote
        subprocess.check_call(["git", "fetch", "origin", current_branch])

        # Get the latest hash for the current branch
        latest_hash = subprocess.check_output(["git", "rev-parse", f"origin/{current_branch}"]).strip().decode("utf-8")

        return latest_hash
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to retrieve latest git hash from remote for the branch: {current_branch}.")


def check_if_ahead_or_behind(current_hash, current_branch):
    """Check if the current commit is ahead or behind the remote branch."""
    try:
        # Get the number of commits ahead or behind
        ahead_behind = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", f"{current_hash}...origin/{current_branch}"]
        ).strip().decode("utf-8")

        ahead, behind = map(int, ahead_behind.split())

        return ahead, behind
    except subprocess.CalledProcessError:
        raise ValueError(f"Failed to determine if the branch {current_branch} is ahead or behind.")
