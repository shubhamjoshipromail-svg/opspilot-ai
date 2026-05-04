"""Create database tables for local development.

Run from the repository root:

    python -m backend.app.db.init_db
"""

from backend.app.database import Base, engine
from backend.app.models import Ticket  # noqa: F401 - imported so metadata registers the model


def init_db() -> None:
    """Create all registered SQLAlchemy tables."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created.")
