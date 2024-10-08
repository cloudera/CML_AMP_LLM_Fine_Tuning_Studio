HF_LOGO = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"
DEFAULT_FTS_GRPC_PORT = "50051"
AXOLOTL_DATASET_FORMAT_CONFIGS_FOLDER_PATH = "ft/config/axolotl/dataset_formats"
AXOLOTL_TRAINING_CONFIGS_TEMPLATE_FILE_PATH = "ft/config/axolotl/training_config/template.yaml"
DEFAULT_CONFIGS = 1
USER_CONFIGS = 0
DATASET_FRACTION_THRESHOLD_FOR_EVALUATION = 90
EVAL_DATASET_DEFAULT_FRACTION = 0.1
BASE_DIR = "./resources/images/icons"
DIVIDER_COLOR = 'green'

# Training Constants
TRAINING_DATA_TEXT_FIELD = "prediction"
TRAINING_DEFAULT_TRAIN_TEST_SPLIT = 0.9
TRAINING_DEFAULT_DATASET_FRACTION = 1.0
TRAINING_DATASET_SEED = 42
EVAL_RANDOM_SEED = 420

# Evaluation constants
EVAL_INPUT_COLUMN = "model_input"
EVAL_OUTPUT_COLUM = "expected_output"
BASE_MODEL_ONLY_IDX = "base-model-only-idx"
BASE_MODEL_ONLY_ADAPTER_ID = "idx-no-adapt"
BASE_MODEL_ONLY_ADAPTER_LOCATION = "no-adapter-location"
USER_DEFINED_IDENTIFIER = "_$useradded$"
# Database constants
DEFAULT_SQLITE_DB_LOCATION = ".app/state.db"
"""
State location of the app. This contains all data that
is a project-specific session.
"""

DEFAULT_PROJECT_DEFAULTS_LOCATION = "data/project_defaults.json"
"""
Default project data defaults location for the application that is populated into
.app/state.db when the AMP is deployed. A user can override the
project defaults JSON when initializing the AMP.
"""


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
