import streamlit as st
from ft.api import *
from ft.consts import IconPaths, DIVIDER_COLOR
from pgs.streamlit_utils import get_fine_tuning_studio_client
fts = get_fine_tuning_studio_client()


def create_header():
    with st.container(border=True):
        col1, col2 = st.columns([1, 17])
        with col1:
            col1.image(IconPaths.AIToolkit.IMPORT_DATASETS)
        with col2:
            col2.subheader('Database Import/Export', divider=DIVIDER_COLOR)
            st.caption(
                "Import database records from JSON files or export your current database to JSON format."
            )


def handle_database_operations():
    export_tab, import_tab = st.tabs(
        ["**Export Database**", "**Import Database**"])

    with export_tab:
        display_export_tab()
    with import_tab:
        display_import_tab()


def display_import_tab():
    st.write("### Import Database from JSON")

    col1, col2 = st.columns([3, 1])
    with col1:
        import_json_path = st.text_input(
            'Database JSON File Location',
            placeholder="path/to/database.json",
            label_visibility="collapsed"
        )

    with col2:
        import_button = st.button("Import", type="primary", use_container_width=True)

    if import_button:
        if import_json_path:
            with st.spinner("Importing database..."):
                try:
                    fts.ImportDatabase(ImportDatabaseRequest(imported_json_path=import_json_path))
                    st.success("Database imported successfully!", icon="✅")
                    st.toast("Import completed successfully.", icon="✅")
                except Exception as e:
                    st.error(f"Failed to import database: {str(e)}", icon="🚨")
                    st.toast(f"Import failed: {str(e)}", icon="🚨")
        else:
            st.error("Please provide an import name!")

    st.info("""
        **Import Guidelines:**

        Your JSON file should follow this structure:
        ```json
        {
            "models": {
                "schema": "CREATE TABLE IF NOT EXISTS models (
                    id VARCHAR NOT NULL,
                    type VARCHAR,
                    framework VARCHAR,
                    name VARCHAR,
                    description VARCHAR,
                    huggingface_model_name VARCHAR,
                    location VARCHAR,
                    cml_registered_model_id VARCHAR,
                    mlflow_experiment_id VARCHAR,
                    mlflow_run_id VARCHAR,
                    PRIMARY KEY (id)
                )",
                "data": [
                    {
                        "id": "f1c3d635-980d-4114-a43e-b3a2eba13910",
                        "type": "huggingface",
                        "framework": null,
                        "name": "bigscience/bloom-1b1",
                        "description": null,
                        "huggingface_model_name": "bigscience/bloom-1b1",
                        "location": null,
                        "cml_registered_model_id": null,
                        "mlflow_experiment_id": null,
                        "mlflow_run_id": null
                    }
                ]
            },
            "datasets": {
                "schema": "CREATE TABLE IF NOT EXISTS datasets (
                    id VARCHAR NOT NULL,
                    type VARCHAR,
                    name VARCHAR,
                    description TEXT,
                    huggingface_name VARCHAR,
                    location TEXT,
                    features TEXT,
                    PRIMARY KEY (id)
                )",
                "data": [
                    {
                        "id": "8ba6f0d8-22a8-4a1a-a823-1dba88f2c377",
                        "type": "huggingface",
                        "name": "philschmid/sql-create-context-copy",
                        "description": "",
                        "huggingface_name": "philschmid/sql-create-context-copy",
                        "location": null,
                        "features": "[\"question\", \"answer\", \"context\"]"
                    }
                ]
            }
        }
        ```

        **Important Notes:**
        - Top level key must be the table name, the next level must be the schema and the data keys.
        - Ensure your JSON file is properly formatted
        - All records should have unique IDs
        - Existing records with the same ID will be ignored.
        - Duplicate recors with different IDs will be imported.
    """)


def display_export_tab():
    st.write("### Export Database to JSON")

    col1, col2 = st.columns([3, 1])

    with col1:
        export_name = st.text_input(
            'Export Name',
            placeholder="Name for the export file",
            label_visibility="collapsed"
        )

    with col2:
        export_button = st.button("Export", type="primary", use_container_width=True)

    if export_button:
        if export_name:
            with st.spinner("Exporting database..."):
                try:
                    exported_file: ExportDatabaseResponse = fts.ExportDatabase(ExportDatabaseRequest())
                    st.success("Database exported successfully!", icon="✅")
                    st.toast("Export completed successfully.", icon="✅")

                    # Add download button after successful export
                    st.download_button(
                        label="Download DB Export",
                        data=exported_file.exported_json,  # Replace with actual exported data
                        file_name=f"{export_name}.json",
                        mime="application/json",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to export database: {str(e)}", icon="🚨")
                    st.toast(f"Export failed: {str(e)}", icon="🚨")
        else:
            st.error("Please provide an export name!")

    st.info("""
        **Export Options:**

        1. Name of the JSON export file


        The exported file will include:
        - Database metadata along with a database schema
        - All records
    """)


create_header()
handle_database_operations()
