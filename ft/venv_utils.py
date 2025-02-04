import sys
import os

def activate_venv(venv_path):
    """
    Activate a Python virtual environment programmatically
    
    Args:
        venv_path: Path to the virtual environment directory
    """
    # Get the activate script path based on OS
    if sys.platform == "win32":
        activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
    else:  # Linux/Mac
        activate_script = os.path.join(venv_path, "bin", "activate")
    
    if not os.path.exists(activate_script):
        raise FileNotFoundError(f"Virtual environment activation script not found at {activate_script}")

    # Add virtual environment's site-packages to sys.path
    site_packages = os.path.join(
        venv_path, 
        "Lib" if sys.platform == "win32" else "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages"
    )
    
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    
    # Set environment variables
    os.environ["VIRTUAL_ENV"] = venv_path
    os.environ["PATH"] = os.pathsep.join([
        os.path.join(venv_path, "Scripts" if sys.platform == "win32" else "bin"),
        os.environ.get("PATH", "")
    ])
    