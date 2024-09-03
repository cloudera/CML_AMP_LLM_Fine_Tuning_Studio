HF_LOGO = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"
DEFAULT_FTS_GRPC_PORT = "50051"
AXOLOTL_DATASET_FORMAT_CONFIGS_FOLDER_PATH = "ft/config/axolotl/dataset_formats"
AXOLOTL_TRAINING_CONFIGS_TEMPLATE_FILE_PATH = "ft/config/axolotl/training_config/template.yaml"
DEFAULT_CONFIGS = 1
USER_CONFIGS = 0

BASE_DIR = "./resources/images/icons"
DIVIDER_COLOR = 'green'


class IconPaths:
    class FineTuningStudio:
        FINE_TUNING_STUDIO = f"{BASE_DIR}/architecture.png"

    class Navigation:
        HOME = f"{BASE_DIR}/home.png"

    class AIToolkit:
        IMPORT_DATASETS = f"{BASE_DIR}/publish.png"
        VIEW_DATASETS = f"{BASE_DIR}/data_object.png"
        IMPORT_BASE_MODELS = f"{BASE_DIR}/neurology.png"
        VIEW_BASE_MODELS = f"{BASE_DIR}/view_day.png"
        CREATE_PROMPTS = f"{BASE_DIR}/chat.png"
        VIEW_PROMPTS = f"{BASE_DIR}/forum.png"

    class Experiments:
        TRAIN_NEW_ADAPTER = f"{BASE_DIR}/forward.png"
        MONITOR_TRAINING_JOBS = f"{BASE_DIR}/subscriptions.png"
        LOCAL_ADAPTER_COMPARISON = f"{BASE_DIR}/difference.png"
        RUN_MLFLOW_EVALUATION = f"{BASE_DIR}/model_training.png"
        VIEW_MLFLOW_RUNS = f"{BASE_DIR}/monitoring.png"

    class CML:
        EXPORT_TO_CML_MODEL_REGISTRY = f"{BASE_DIR}/move_group.png"
        PROJECT_OWNER = f"{BASE_DIR}/account_circle.png"
