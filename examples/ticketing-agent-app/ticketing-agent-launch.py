
import os

if os.getenv("IS_COMPOSABLE", "") != "":
  os.chdir("/home/cdsw/fine-tuning-studio")

!uv run -m streamlit run examples/ticketing-agent-app/ticketing-agent-app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1