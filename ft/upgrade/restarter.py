import os
import cmlapi
import streamlit as st

def restart_application_function():
    cml = cmlapi.default_client()
    project_id = os.getenv("CDSW_PROJECT_ID")
    apps_list = cml.list_applications(project_id).applications
    found_app_list = list(filter(lambda app: 'Fine Tuning Studio' in app.name, apps_list))
    if len(found_app_list) > 0:
        app = found_app_list[0]
        if app.status == "APPLICATION_RUNNING":
            try:
                cml.restart_application(project_id, app.id)
                st.success(f"Application {app.name} restarted successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to restart application {app.name}: {str(e)}")
                return False
        else:
            st.error(f"Application {app.name} is not running.")
            return False