import streamlit as st
from ft.utils import get_env_variable
from ft.consts import IconPaths
from streamlit_navigation_bar import st_navbar
import os
# Module for custom CSS

if os.getenv("IS_COMPOSABLE", "") != "":
    def apply_custom_css():
        css = '''
        <style>
            h3 {
                font-size: 1.1rem;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 0.9rem;
            }
        </style>
        '''
        st.markdown(css, unsafe_allow_html=True)

    # Module for setting up navigation


    def setup_navigation():
        pg = st.navigation([
            st.Page("pgs/home.py", title="Home"),
            st.Page("pgs/database.py", title="Database Import and Export"),
            st.Page("pgs/datasets.py", title="Import Datasets"),
            st.Page("pgs/view_datasets.py", title="View Datasets"),
            st.Page("pgs/models.py", title="Import Base Models"),
            st.Page("pgs/view_models.py", title="View Base Models"),
            st.Page("pgs/prompts.py", title="Create Prompts"),
            st.Page("pgs/view_prompts.py", title="View Prompts"),
            st.Page("pgs/train_adapter.py", title="Train a New Adapter"),
            st.Page("pgs/jobs.py", title="Training Job Tracking"),
            st.Page("pgs/evaluate.py", title="Local Adapter Comparison"),
            st.Page("pgs/mlflow.py", title="Run MLFlow Evaluation"),
            st.Page("pgs/mlflow_jobs.py", title="View MLflow Runs"),
            st.Page("pgs/export.py", title="Export And Deploy Model"),
            st.Page("pgs/sample_ticketing_agent_app_embed.py", title="Sample Ticketing Agent App"),
            st.Page("pgs/feedback.py", title="Feedback"),
        ], position="hidden")
        # pg.run()
        return pg



    # Main function to orchestrate the setup
    st.set_page_config(
        page_title="Fine Tuning Studio",
        page_icon=IconPaths.FineTuningStudio.FINE_TUNING_STUDIO,
        layout="wide"
    )

    styles = {
        "nav": {
            "background-color": "white",
            "display": "flex",
            
            "height": ".01rem"
        },
    }
    options = {
        'show_menu': False
    }
    pages = ['Home']
    page = st_navbar(pages,
                        styles=styles,
                        options=options
                        )

    # Include Material Icons stylesheet
    st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">', unsafe_allow_html=True)

    # Bootstrap CSS for styling
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    # Custom CSS for bigger dropdown menus and icons for links
    st.markdown("""
        <style>
        .navbar {
            background-color: #132329;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
            padding: 10px;
            margin: 0;
            left: 0;
            display: flex;
            justify-content: flex-start;
            gap: 20px; /* Control space between navbar items */
        }
        
        .navbar a {
            color: white;
            padding: 10px 20px; /* Adjust padding to control spacing around each item */
            text-decoration: none;
            font-size: 16px;
            display: inline-block;
        }
        
        .navbar a:hover {
            background-color: #2980B9;
        }
        
        .dropdown {
            position: relative;
            display: inline-block;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: #f9f9f9;
            min-width: 325px; /* Increased width of the dropdown by 30% (from 250px to 325px) */
            padding: 13px 0; /* Increased padding for dropdown items */
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
        }
        .dropdown-content a {
            color: black;
            padding: 14px 25px; /* Increased padding to make dropdown items bigger */
            text-decoration: none;
            display: block;
            text-align: left;
        }
        .dropdown-content a:hover {
            background-color: #ddd;
        }
        .dropdown:hover .dropdown-content {
            display: block;
        }
        .dropdown:hover .dropbtn {
            background-color: #2980B9;
        }
        .material-icons {
            font-size: 18px;
            vertical-align: middle;
            margin-right: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Navbar HTML with bigger dropdown menus and icons for each link
    st.markdown("""
    <nav class="navbar">
    <a href="/home" target="_self"><span class="material-icons">home</span>Fine Tuning Studio</a>
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">build</span> Database Import Export</a>
        <div class="dropdown-content">  
        <a href="/database" target="_self"><span class="material-icons">publish</span>Import and Export Application Database</a>
        </div>
    </div>
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">build</span> Resources</a>
        <div class="dropdown-content">
        <a href="/datasets" target="_self"><span class="material-icons">publish</span> Import Datasets</a>
        <a href="/view_datasets" target="_self"><span class="material-icons">data_object</span> View Datasets</a>
        <a href="/models" target="_self"><span class="material-icons">download</span> Import Base Models</a>
        <a href="/view_models" target="_self"><span class="material-icons">view_day</span> View Base Models</a>
        <a href="/prompts" target="_self"><span class="material-icons">chat</span> Create Prompts</a>
        <a href="/view_prompts" target="_self"><span class="material-icons">forum</span> View Prompts</a>
        </div>
    </div>
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">science</span> Experiment</a>
        <div class="dropdown-content">
        <a href="/train_adapter" target="_self"><span class="material-icons">forward</span> Train a New Adapter</a>
        <a href="/jobs" target="_self"><span class="material-icons">subscriptions</span> Monitor Training Jobs</a>
        <a href="/evaluate" target="_self"><span class="material-icons">difference</span> Local Adapter Comparison</a>
        <a href="/mlflow" target="_self"><span class="material-icons">model_training</span> Run MLFlow Evaluation</a>
        <a href="/mlflow_jobs" target="_self"><span class="material-icons">dashboard</span> View MLflow Runs</a>
        </div>
    </div>
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">cloud</span> AI Workbench</a>
        <div class="dropdown-content">
        <a href="/export" target="_self"><span class="material-icons">upgrade</span> Export And Deploy Model</a>
        </div>
    </div>
    
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">subscriptions</span> Examples</a>
        <div class="dropdown-content">
        <a href="/sample_ticketing_agent_app_embed" target="_self"><span class="material-icons">forward</span> Ticketing Agent App</a>
        </div>
    </div>
    
    <div class="dropdown">
        <a class="dropbtn" href="#" target="_self"><span class="material-icons">subscriptions</span> Feedback</a>
        <div class="dropdown-content">
        <a href="/feedback" target="_self"><span class="material-icons">forward</span> Provide Feedback</a>
        </div>
    </div>
    </nav>
    """, unsafe_allow_html=True)

    apply_custom_css()
    pg = setup_navigation()
    pg.run()

else:

    def apply_custom_css_sidebar():
        css = '''
        <style>
            h3 {
                font-size: 1.1rem;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 0.9rem;
            }
            [data-testid="stHeader"] {
                color: #4CAF50;
                background-color: #f0f0f0;
            }
            [data-testid="stSidebarContent"] {
                background: #16262c;
                color: white;
            }
            [data-testid="stSidebarContent"] * {
                color: white;
            }
        </style>
        '''
        st.markdown(css, unsafe_allow_html=True)

    # Module for setting up navigation


    def setup_navigation_sidebar():
        pg = st.navigation([
            st.Page("pgs/home.py", title="Home"),
            st.Page("pgs/database.py", title="Database Import and Export"),
            st.Page("pgs/datasets.py", title="Import Datasets"),
            st.Page("pgs/view_datasets.py", title="View Datasets"),
            st.Page("pgs/models.py", title="Import Base Models"),
            st.Page("pgs/view_models.py", title="View Base Models"),
            st.Page("pgs/prompts.py", title="Create Prompts"),
            st.Page("pgs/view_prompts.py", title="View Prompts"),
            st.Page("pgs/train_adapter.py", title="Train a New Adapter"),
            st.Page("pgs/jobs.py", title="Training Job Tracking"),
            st.Page("pgs/evaluate.py", title="Local Adapter Comparison"),
            st.Page("pgs/mlflow.py", title="Run MLFlow Evaluation"),
            st.Page("pgs/mlflow_jobs.py", title="View MLflow Runs"),
            st.Page("pgs/export.py", title="Export And Deploy Model"),
            st.Page("pgs/sample_ticketing_agent_app_embed.py", title="Sample Ticketing Agent App"),
            st.Page("pgs/feedback.py", title="Feedback"),
        ], position="hidden")
        pg.run()

    # Module for sidebar content


    def setup_sidebar():
        with st.sidebar:
            st.image("./resources/images/ft-logo.png")
            st.subheader("")
            st.markdown("Navigation")
            st.page_link("pgs/home.py", label="Home", icon=":material/home:")
            st.page_link("pgs/database.py", label="Database Import and Export", icon=":material/database:")
            st.write("\n")

            st.markdown("Resources")
            st.page_link("pgs/datasets.py", label="Import Datasets", icon=":material/publish:")
            st.page_link("pgs/view_datasets.py", label="View Datasets", icon=":material/data_object:")
            st.page_link("pgs/models.py", label="Import Base Models", icon=":material/neurology:")
            st.page_link("pgs/view_models.py", label="View Base Models", icon=":material/view_day:")
            st.page_link("pgs/prompts.py", label="Create Prompts", icon=":material/chat:")
            st.page_link("pgs/view_prompts.py", label="View Prompts", icon=":material/forum:")
            st.write("\n")

            st.markdown("Experiments")
            st.page_link("pgs/train_adapter.py", label="Train a New Adapter", icon=":material/forward:")
            st.page_link("pgs/jobs.py", label="Monitor Training Jobs", icon=":material/subscriptions:")
            st.page_link("pgs/evaluate.py", label="Local Adapter Comparison", icon=":material/difference:")
            st.page_link("pgs/mlflow.py", label="Run MLFlow Evaluation", icon=":material/model_training:")
            st.page_link("pgs/mlflow_jobs.py", label="View MLflow Runs", icon=":material/monitoring:")
            st.write("\n")

            st.markdown("AI Workbench")
            st.page_link("pgs/export.py", label="Export And Deploy Model", icon=":material/move_group:")
            st.write("\n")

            st.markdown("Examples")
            st.page_link("pgs/sample_ticketing_agent_app_embed.py", label="Ticketing Agent App", icon=":material/deployed_code:")
            st.markdown("Feedback")
            st.page_link("pgs/feedback.py", label="Feedback", icon=":material/feedback:")
            st.subheader("", divider="green")

            project_owner = get_env_variable('PROJECT_OWNER', 'User')
            cdsw_url = get_env_variable('CDSW_DOMAIN', 'CDSW Url')
            st.page_link(f"https://{cdsw_url}", label=f"{project_owner}", icon=":material/account_circle:")


    # Main function to orchestrate the setup
    st.set_page_config(
        page_title="Fine Tuning Studio",
        page_icon=IconPaths.FineTuningStudio.FINE_TUNING_STUDIO,
        layout="wide"
    )
    apply_custom_css_sidebar()
    setup_navigation_sidebar()
    setup_sidebar()
