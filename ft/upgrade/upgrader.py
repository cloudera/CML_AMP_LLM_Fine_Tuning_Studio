import streamlit as st
import subprocess
import sys
import importlib
import ensurepip
from time import sleep

def run_git_pull():
    """
    Execute git pull to update the repository.
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Fetch the latest changes and pull
        result = subprocess.run(
            ['git', 'pull'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        st.success("Git pull successful!")
        st.text(result.stdout)
        print("Git pull successful! Sleeping for 5 seconds.")
        sleep(5)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Git pull failed: {e.stderr}")
        return False



def is_package_installed(package_name):
    """
    Check if a Python package is installed.
    
    Args:
        package_name (str): Name of the package to check
    
    Returns:
        bool: True if package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """
    Install a Python package using pip.
    
    Args:
        package_name (str): Name of the package to install
    
    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        # Ensure pip is available
        ensurepip.bootstrap()
        
        # Install the package
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        st.success(f"{package_name} installed successfully!")
        st.text(result.stdout)
        return True
    except (subprocess.CalledProcessError, Exception) as e:
        st.error(f"Failed to install {package_name}: {str(e)}")
        return False

def run_alembic_upgrade():
    """
    Run Alembic database migrations after checking and installing if necessary.
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Check if Alembic is installed
        if not is_package_installed('alembic'):  # Check if a particular version is needed here
            st.warning("Alembic not found. Attempting to install...")
            
            # Try to install Alembic
            install_result = install_package('alembic')
            
            if not install_result:
                st.error("Could not install Alembic. Upgrade process cannot continue.")
                return False
        
        # Verify installation after attempted install
        if not is_package_installed('alembic'):
            st.error("Alembic installation failed unexpectedly.")
            return False
        
        # Run Alembic upgrade
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        st.success("Alembic database upgrade successful!")
        st.text(result.stdout)
        print("Alembic upgrade successful! Sleeping for 10 seconds.")
        sleep(10)
        return True
    
    except subprocess.CalledProcessError as e:
        st.error(f"Alembic upgrade failed: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error during Alembic upgrade: {str(e)}")
        return False
