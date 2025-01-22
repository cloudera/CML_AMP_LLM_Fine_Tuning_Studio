import os

if os.getenv("IS_COMPOSABLE", "") != "":
    !uv run /home/cdsw/fine-tuning-studio/bin/initialize-project-defaults.py
else:
    !uv run /home/cdsw/bin/initialize-project-defaults.py
