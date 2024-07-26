import streamlit as st
from ft.state import get_state
from st_aggrid import AgGrid
from st_aggrid import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import os
import requests

# Container for the layout
with st.container(border=True):
    col1, col2 = st.columns([1, 17])
    with col1:
        col1.image("./resources/images/subscriptions_24dp_EA3323_FILL0_wght400_GRAD0_opsz40.png")
    with col2:
        col2.subheader('Monitor Training Jobs', divider='red')
        st.caption("Monitor your fine-tuning jobs, track progress, and ensure optimal performance throughout the training process.")

st.write("\n\n")

# Fetch current jobs from state
current_jobs = get_state().jobs

# Check if there are any current jobs
if not current_jobs:
    st.info("No fine-tuning jobs triggered.", icon=":material/info:")
else:
    # Convert jobs to DataFrame
    jobs_df = pd.DataFrame([res.model_dump() for res in current_jobs])

    # Check if 'cml_job_id' exists in jobs_df
    if 'cml_job_id' not in jobs_df.columns:
        st.error("Column 'cml_job_id' not found in jobs_df")
    else:
        # Get project URL from environment variables
        project_url = os.getenv('CDSW_PROJECT_URL', default='CDSW_PROJECT_URL')
        project_url = project_url.replace('/api/v1/projects', '')

        # Use APIv1 for jobs to get id and URL for each job
        HOST = os.getenv('CDSW_PROJECT_URL')
        API_KEY = os.getenv('CDSW_API_KEY')

        url = "/".join([HOST, "jobs"])
        res = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            auth=(API_KEY, "")
        )

        if res.status_code != 200:
            st.error("Failed to fetch jobs from API")
        else:
            # Convert API response to DataFrame
            cml_jobs_list = res.json()
            cml_jobs_list_df = pd.DataFrame(cml_jobs_list)

            # Check if 'public_identifier' exists in cml_jobs_list_df
            if 'public_identifier' not in cml_jobs_list_df.columns:
                st.error("Column 'public_identifier' not found in cml_jobs_list_df")
            else:
                # Merge the DataFrames for app state jobs and apiv1 cml jobs
                display_df = pd.merge(jobs_df, cml_jobs_list_df, left_on='cml_job_id', right_on='public_identifier')

                # Filter for only columns we care about
                display_df = display_df[['job_id', 'html_url', 'adapter_id', 'base_model_id', 'dataset_id', 'prompt_id', 'num_workers', 'latest']]

                # Apply status color renderer
                display_df['latest'] = display_df['latest'].apply(lambda x: x['status'])

                # Build Grid Options
                gd = GridOptionsBuilder.from_dataframe(display_df)

                # Renderer for links
                cell_url_renderer = JsCode("""
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
                gd.configure_column("html_url", headerName="CML Job Link", cellRenderer=cell_url_renderer, width=300)

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
                gridoptions = gd.build()

                # Display the grid with the merged and filtered dataframe
                AgGrid(display_df, gridOptions=gridoptions, enable_enterprise_modules=False, allow_unsafe_jscode=True, height=500, theme='alpine')
