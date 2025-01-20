import os

if os.getenv("IS_COMPOSABLE", "") != "":
    !uv run -m streamlit run /home/cdsw/fine-tuning-studio/examples/ticketing-agent-app/ticketing-agent-app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
else:
    !uv run -m streamlit run /home/cdsw/examples/ticketing-agent-app/ticketing-agent-app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1