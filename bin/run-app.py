import subprocess
import os 

if os.getenv("IS_COMPOSABLE", "") != "":
  os.chdir("/home/cdsw/fine-tuning-studio")

CDSW_APP_PORT = os.environ.get("CDSW_APP_PORT")
out = subprocess.run([f"bash ./bin/start-app-script.sh {CDSW_APP_PORT}"], shell=True, check=True)
print(out)

print("App start script is complete.")