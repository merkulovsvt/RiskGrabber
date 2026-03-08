from collections.abc import AsyncGenerator
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import reflection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, scoped_session, sessionmaker

from .config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

async_engine = create_async_engine(
    settings.get_async_database_url(),
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db_sync() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_db() -> None:
    from . import models  # noqa: F401 — avoid circular import

    Base.metadata.create_all(bind=engine)

    try:
        insp = reflection.Inspector.from_engine(engine)
        if "review_risks" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("review_risks")]
            if "review_date" not in cols:
                col_type = "DATETIME" if engine.dialect.name == "sqlite" else "TIMESTAMP"
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE review_risks ADD COLUMN review_date {col_type}"))
                    conn.commit()
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE review_risks SET review_date = (
                        SELECT COALESCE(r.published_at, r.scraped_at) FROM reviews r
                        WHERE r.id = review_risks.review_id
                    ) WHERE review_date IS NULL
                """))
                conn.commit()
    except Exception:
        pass

    try:
        insp = reflection.Inspector.from_engine(engine)
        if "reviews" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("reviews")]
            if "criticality_score" not in cols:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE reviews ADD COLUMN criticality_score INTEGER"))
                    conn.commit()
    except Exception:
        pass

    try:
        insp = reflection.Inspector.from_engine(engine)
        if "risks" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("risks")]
            text_type = "TEXT" if engine.dialect.name == "sqlite" else "TEXT"
            # risk_factors — список строк: в PostgreSQL JSONB, в SQLite TEXT (JSON-строка)
            risk_factors_type = "JSONB" if engine.dialect.name == "postgresql" else "TEXT"
            with engine.connect() as conn:
                if "description" not in cols:
                    conn.execute(text(f"ALTER TABLE risks ADD COLUMN description {text_type} DEFAULT ''"))
                    conn.commit()
                if "risk_factors" not in cols:
                    conn.execute(text(f"ALTER TABLE risks ADD COLUMN risk_factors {risk_factors_type}"))
                    conn.commit()
    except Exception:
        pass

    try:
        insp = reflection.Inspector.from_engine(engine)
        if "risks" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("risks")]
            if "severity" in cols:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE risks DROP COLUMN severity"))
                    conn.commit()
    except Exception:
        pass

    try:
        insp = reflection.Inspector.from_engine(engine)
        if "risks" in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns("risks")]
            if "title" in cols:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE risks DROP COLUMN title"))
                    conn.commit()
    except Exception:
        pass

