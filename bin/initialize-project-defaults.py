from ft.db.utils import import_defaults
from ft.db.dao import FineTuningStudioDao, get_sqlite_db_location, delete_database
import os
import json
from alembic import command
from alembic.config import Config

def main():
    
    # Delete the existing database information. This shouldn't exist 
    # in this AMP's current deployment pattern, because this script typically
    # only runs once at AMP initialization and then does not run again. At AMP
    # initialization the project is not populated yet, so there is not an .app/state.db
    # file yet.
    if os.path.exists(get_sqlite_db_location()):
        delete_database()

    # Make the app state location if it's not yet created.
    os.makedirs(os.path.dirname(get_sqlite_db_location()), exist_ok=True)

    # Import project defaults.
    import_defaults()
    
    return


def run_migrations():
    """
    Run database migrations using Alembic.
    

    """
    db_path = get_sqlite_db_location()
    # Create Alembic configuration
    alembic_cfg = Config('alembic.ini')
    
    # Set the SQLAlchemy database URL in the Alembic configuration
    alembic_cfg.set_main_option('sqlalchemy.url', f'sqlite:///{db_path}')
    
    try:
        # Upgrade to the latest database schema
        command.upgrade(alembic_cfg, 'head')
        print(f"Migrations completed for {db_path}")
    except Exception as e:
        print(f"Migration error: {e}")


if __name__ == "__main__":
    main()
