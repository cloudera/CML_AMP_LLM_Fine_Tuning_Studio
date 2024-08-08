import mlflow
import os


class AxolotlFineTuner:
    # Initialize the fine-tuner with a specific MLflow experiment
    # Default: bigscience/bloom-1b1
    def __init__(self, ft_job_uuid=""):
        # Set the MLflow experiment
        if not ft_job_uuid:
            raise ValueError("Fine-tuning job UUID must be provided.")
        mlflow.set_experiment(ft_job_uuid)

    # Train/Fine-tune the model with SFTTrainer and a provided dataset
    def train(self, yaml_file_path: str):
        # Ensure the YAML file path is provided
        if not yaml_file_path:
            raise ValueError("YAML file path must be provided for training with Axolotl.")

        # Check if the YAML file exists
        if not os.path.exists(yaml_file_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")

        # Construct the Axolotl training command
        command = f"accelerate launch -m axolotl.cli.train {yaml_file_path}"
        print(f"Executing command: {command}")

        # Execute the command and check for errors
        exit_code = os.system(command)
        if exit_code != 0:
            raise RuntimeError(f"Training command failed with exit code {exit_code}")

        print("Training with Axolotl Complete!")

# Example usage with edge case handling
# try:
#     fine_tuner = AxolotlFineTuner(ft_job_uuid="example_experiment_uuid")
#     fine_tuner.train(yaml_file_path="path_to_yaml_file.yaml")
# except (ValueError, FileNotFoundError, RuntimeError) as e:
#     print(f"Error: {e}")
