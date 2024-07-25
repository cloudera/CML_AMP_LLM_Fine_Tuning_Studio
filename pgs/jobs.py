import streamlit as st
from ft.state import get_state
from st_aggrid import AgGrid
from st_aggrid import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import os

with st.container(border=True):
    col1, col2 = st.columns([1,13])
    with col1:
        col1.image("./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Monitor Training Jobs', divider='orange')
        st.write("Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")

st.write("\n\n")

current_jobs = get_state().jobs

jobs_df = pd.DataFrame([res.model_dump() for res in current_jobs])
project_url = os.getenv('CDSW_PROJECT_URL', default='CDSW_PROJECT_URL')
project_url = project_url.replace('/api/v1/projects', '')

# Use APIv1 for jobs
# This is needed to get id and URL for each job
import requests
import json

HOST = os.getenv('CDSW_PROJECT_URL')

API_KEY = os.getenv('CDSW_API_KEY')

url = "/".join([HOST,"jobs"])
res = requests.get(
    url,
    headers = {"Content-Type": "application/json"},
    auth = (API_KEY,"")
)

cml_jobs_list = res.json()
cml_jobs_list_df = pd.DataFrame(cml_jobs_list)

# Merge the dataframes for app state jobs and apiv1 cml jobs
# TODO: cut out columns
# TODO: mrege in experiments url
display_df = pd.merge(jobs_df, cml_jobs_list_df, left_on='cml_job_id', right_on='public_identifier')

# Filter for only columns we care about
display_df = display_df[['job_id','html_url','adapter_id','base_model_id','dataset_id','prompt_id','num_workers', 'latest']]

#
display_df['latest'] = display_df['latest'].apply(lambda x: x['status'])


# Build Grid Options
gd=GridOptionsBuilder.from_dataframe(display_df)

# Renderer for links
cell_url_renderer=JsCode("""
        class UrlCellRenderer {
          init(params) {
            this.eGui = document.createElement('a');
            this.eGui.innerText = 'CML Job';
            this.eGui.setAttribute('href', params.value);
            this.eGui.setAttribute('style', "text-decoration:underline;");
            this.eGui.setAttribute('target', "_blank");
          }
          getGui() {
            return this.eGui;
          }
        }
    """)
gd.configure_column("html_url", headerName="CML Job Link", cellRenderer=cell_url_renderer,
                width=300)

# Renderer for status colors
cell_status_renderer = JsCode("""
    function(params) {
        if (params.value == 'succeeded') {
            return {
                'color': 'white',
                'backgroundColor': 'green'
            }
        } else if (params.value == 'running') {
            return {
                'color': 'white',
                'backgroundColor': 'blue'
            }
        } else if (params.value == 'scheduling') {
            return {
                'color': 'black',
                'backgroundColor': 'grey'
            }
        } else {
            return {
                'color': 'black',
                'backgroundColor': 'red'
            }
        }
    };
    """) 
    
gd.configure_column("latest", cellStyle=cell_status_renderer)

# Build all grid options
gridoptions=gd.build()

# Display the grid with sdisplay_df (merged and filtered dataframe)
grid_return = AgGrid(display_df, gridOptions=gridoptions, enable_enterprise_modules=False, allow_unsafe_jscode=True, height=500, theme='alpine')


