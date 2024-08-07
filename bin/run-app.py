import subprocess
import os 

CDSW_APP_PORT = os.environ.get("CDSW_APP_PORT")
out = subprocess.run([f"bash ./bin/start-app-script.sh {CDSW_APP_PORT}"], shell=True, check=True)
print(out)

print("App start script is complete.")