import os

from ft.db.model import Base

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager


DEFAULT_SQLITE_DB_LOCATION = ".app/state.db"
"""
State location of the app. This contains all data that
is a project-specific session.
"""


def get_sqlite_db_location():
    """
    Get the location of the currently loaded state file.
    """
    if os.environ.get("FINE_TUNING_STUDIO_SQLITE_DB"):
        return os.environ.get("FINE_TUNING_STUDIO_SQLITE_DB")
    return DEFAULT_SQLITE_DB_LOCATION


class FineTuningStudioDao():
    """
    Data access layer for the Fine Tuning Studio application. In the future,
    this should be abstracted out to a base DAO class with different implementations
    depending on the underlying SQL engine, if necessary. However given that we don't
    yet know the necessary level of abstraction, we will air on the side of code
    simplicity and not build the base class yet.
    """

    def __init__(self, engine_url: str = None, echo: bool = False, engine_args: dict = {}):
        if engine_url is None:
            engine_url = f"sqlite+pysqlite:///{get_sqlite_db_location()}"

        self.engine = create_engine(
            engine_url,
            echo=echo,
            **engine_args,
        )
        self.Session = sessionmaker(bind=self.engine, autoflush=True, autocommit=False)

        # Create all of our required tables if they do not yet exist.
        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self):
        """
        Provides a context manager for a session that automatically
        attempts a session commit after completion of the context, and will
        automatically rollback if there are failures, and finally will close
        the session once complete, releasing the sesion back to the session pool.
        """
        session = self.Session()
        try:
            yield session
            session.commit()  # Commit on successful operation
        except Exception as e:
            session.rollback()  # Rollback in case of error
            raise e
        finally:
            session.close()
