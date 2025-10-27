"""Database session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for getting DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
