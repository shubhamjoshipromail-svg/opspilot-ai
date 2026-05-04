import os

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    DATABASE_URL: str
    APP_ENV: str = "development"

    @property
    def database_url(self) -> str:
        """Lowercase alias for call sites that prefer Python-style names."""
        return self.DATABASE_URL


def _normalize_database_url(database_url: str) -> str:
    """Normalize provider URLs only when SQLAlchemy needs the explicit dialect."""
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


load_dotenv()

settings = Settings(
    DATABASE_URL=_normalize_database_url(
        os.getenv("DATABASE_URL", "sqlite:///./opspilot_dev.db")
    ),
    APP_ENV=os.getenv("APP_ENV", "development"),
)
