import os

if os.getenv("IS_COMPOSABLE", "") != "":
  os.chdir("/home/cdsw/fine-tuning-studio")

!uv run bin/initialize-project-defaults.py
