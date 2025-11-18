"""Database configuration."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional

Base = declarative_base()

# Will be configured at runtime
_engine = None
_SessionLocal = None


def init_db(database_url: str):
    """Initialize database connection.

    Args:
        database_url: PostgreSQL connection string
                     e.g., postgresql://user:pass@localhost/alpha_signal
    """
    global _engine, _SessionLocal
    _engine = create_engine(database_url, echo=False)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_db_session():
    """Get database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()


def create_tables():
    """Create all tables."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    Base.metadata.create_all(bind=_engine)
