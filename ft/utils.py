import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import os
import requests
import torch
from huggingface_hub import login


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
            print("Attempting HF Login...")
            print("HF token from arguments: ", access_token)
            login(access_token)
        else:
            print("HF Access token not provided. Cannot use gated models.")
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
