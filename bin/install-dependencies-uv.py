import subprocess
import sys
from pathlib import Path
import os

DEFAULT_BASE_PATH="/home/cdsw/fine-tuning-studio"

def change_base_path():
    root_dir = DEFAULT_BASE_PATH if os.getenv("IS_COMPOSABLE", "") != "" else "/home/cdsw"
    print(f"The root directory is {root_dir}")
    os.chdir(root_dir)


def check_uv_installation():
    """Verify if uv is installed in the system."""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False
    
def install_uv():
    """Install uv if not installed."""
    try:
        subprocess.run(['pip', 'install', 'uv'], check=True)
        print("✓ Installed uv")
    except subprocess.CalledProcessError as e:
        print(f"Error installing uv: {e}")
        sys.exit(1)


def migrate_requirements(requirements_path =None):
    """Migrate requirements.txt to uv.txt and validate packages."""
    try:

        # Validate and compile dependencies using uv
        result = subprocess.run(
            ['uv', 'add', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Successfully migrated requirements to UV format")
            return True
        else:
            print(f"Error compiling requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error during migration: {e}")
        return False



def main():
    print("Changing the base path if composability is enabled")
    change_base_path()
    print("Base path changed successfully")
    if not check_uv_installation():
        print("UV is not installed. PTrying to install uv")
        try:
            install_uv()
        except:
            sys.exit(1)

    project_path = Path.cwd()
    requirements_path = project_path / 'requirements.txt'

    if not requirements_path.exists():
        print("Error: requirements.txt not found")
        sys.exit(1)

    print("\nStarting UV migration process...")
    
    if migrate_requirements():
            print("\nMigration completed successfully! 🎉")
    else:
        print("\nMigration failed. Please check the errors above.")

if __name__ == "__main__":
    main()