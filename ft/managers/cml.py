import cmlapi
from cmlapi import CMLServiceApi
import os


class CMLManager:
    """
    Simple class for collecting CML Workspace, project
    and engine-specific environment variables, and other
    global components.
    """

    def __init__(self):
        """
        Collect basic environment variables and generate a
        default CML API client.
        """
        self.cml_api_client: CMLServiceApi = cmlapi.default_client()
        self.client = self.cml_api_client # backwards compatibility
        
        # Load in environment variables
        self.project_id = os.getenv("CDSW_PROJECT_ID")
        self.engine_id = os.getenv("CDSW_ENGINE_ID")
        self.master_id = os.getenv("CDSW_MASTER_ID")
        self.master_ip = os.getenv("CDSW_MASTER_IP")
        self.domain = os.getenv("CDSW_DOMAIN")
