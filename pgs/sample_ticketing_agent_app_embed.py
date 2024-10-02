import streamlit.components.v1 as components
import streamlit as st
import os
import cmlapi

cml = cmlapi.default_client()
project_id = os.getenv("CDSW_PROJECT_ID")
apps_list = cml.list_applications(project_id).applications

found_app_list = list(filter(lambda app: 'Sample-Ticketing-Agent-Application' in app.name, apps_list))
if len(found_app_list) > 0:
    app = found_app_list[0]
    if app.status == "APPLICATION_RUNNING":
        subdomain = app.subdomain
        url = "https://" + subdomain + "." + os.getenv("CDSW_DOMAIN")
        components.iframe(url, height=1600)
    else:
        st.error(f"Application [Sample-Ticketing-Agent-Application] not started.\nPlease restart the application in the Workspace Project")
else:
    st.error(f"Application [Sample-Ticketing-Agent-Application] not found.\nPlease refer to `examples/ticketing-agent-app/README.md` for creation.")