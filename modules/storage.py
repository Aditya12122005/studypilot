"""Optional PostgreSQL persistence for StudyPilot user data."""

import hashlib
import io
import os
import re
import secrets
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np


class StorageError(RuntimeError):
    """A persistence error with a safe message for display in the app."""


def get_database_url():
    """Return the configured database URL without logging or displaying it."""
    value = os.environ.get("DATABASE_URL", "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get("DATABASE_URL", "")
    except Exception:
        return ""
    return value.strip() if isinstance(value, str) else ""


def new_user_id():
    """Create a high-entropy recovery ID used to access one user's data."""
    return secrets.token_urlsafe(32)


def _user_key(user_id):
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 256:
        raise StorageError("Enter a valid private data ID.")
    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()


def _normalize_database_url(database_url):
    url = database_url.strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgresql+psycopg2://"):
        url = "postgresql+psycopg://" + url[len("postgresql+psycopg2://") :]
    if not url.startswith("postgresql+psycopg://"):
        raise StorageError("Optional persistence requires a PostgreSQL DATABASE_URL.")
    return url


@lru_cache(maxsize=2)
def _database(database_url):
    """Create a quiet SQLAlchemy engine and initialize the small schema."""
    engine = None
    try:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        engine = sa.create_engine(
            _normalize_database_url(database_url),
            echo=False,
            hide_parameters=True,
            pool_pre_ping=True,
        )
        metadata = sa.MetaData()
        documents = sa.Table(
            "studypilot_user_documents",
            metadata,
            sa.Column("user_id", sa.String(64), primary_key=True),
            sa.Column("document_id", sa.String(64), primary_key=True),
            sa.Column("document_name", sa.Text, nullable=False),
            sa.Column("document_text", sa.Text, nullable=False),
            sa.Column("chunks", sa.JSON, nullable=False),
            sa.Column("embeddings", sa.LargeBinary, nullable=False),
            sa.Column("flashcards", sa.JSON),
            sa.Column("flashcard_progress", sa.JSON),
            sa.Column("quiz", sa.JSON),
            sa.Column("quiz_results", sa.JSON),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        daily_usage = sa.Table(
            "studypilot_daily_usage",
            metadata,
            sa.Column("user_id", sa.String(64), primary_key=True),
            sa.Column("usage_date", sa.Date, primary_key=True),
            sa.Column("call_count", sa.Integer, nullable=False),
        )
        metadata.create_all(engine)
        return engine, sa, pg_insert, documents, daily_usage
    except Exception:
        if engine is not None:
            engine.dispose()
        raise StorageError("Could not initialize the configured persistence database.") from None


def _get_database():
    database_url = get_database_url()
    if not database_url:
        raise StorageError("Database persistence is not configured.")
    return _database(database_url)


def _document_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise StorageError("The document identifier is invalid.")
    return value


def _normalize_chunks(chunks):
    normalized = []
    for chunk in chunks:
        if isinstance(chunk, str):
            normalized.append(
                {"text": chunk, "page_start": None, "page_end": None}
            )
        elif isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
            normalized.append(
                {
                    "text": chunk["text"],
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
            )
        else:
            raise StorageError("The document chunks could not be stored.")
    return normalized


def _encode_embeddings(embeddings, chunk_count):
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != chunk_count:
        raise StorageError("The document embeddings could not be stored.")
    if not np.isfinite(array).all():
        raise StorageError("The document embeddings could not be stored.")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _decode_embeddings(blob, chunk_count):
    try:
        array = np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception:
        raise StorageError("Saved document data could not be loaded.") from None
    if array.ndim != 2 or array.shape[0] != chunk_count:
        raise StorageError("Saved document data could not be loaded.")
    return np.asarray(array, dtype=np.float32)


def save_document(user_id, document_id, document_name, document_text, chunks, embeddings):
    """Save one document, page-aware chunks, and its embedding array."""
    owner = _user_key(user_id)
    document_id = _document_id(document_id)
    if not isinstance(document_text, str) or not isinstance(document_name, str):
        raise StorageError("The document could not be stored.")
    chunks = _normalize_chunks(chunks)
    embedding_blob = _encode_embeddings(embeddings, len(chunks))
    try:
        engine, _, pg_insert, documents, _ = _get_database()
        values = {
            "user_id": owner,
            "document_id": document_id,
            "document_name": document_name[:512],
            "document_text": document_text,
            "chunks": chunks,
            "embeddings": embedding_blob,
            "updated_at": datetime.now(timezone.utc),
        }
        statement = pg_insert(documents).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[documents.c.user_id, documents.c.document_id],
            set_={
                "document_name": statement.excluded.document_name,
                "document_text": statement.excluded.document_text,
                "chunks": statement.excluded.chunks,
                "embeddings": statement.excluded.embeddings,
                "updated_at": statement.excluded.updated_at,
            },
        )
        with engine.begin() as connection:
            connection.execute(statement)
    except StorageError:
        raise
    except Exception:
        raise StorageError("Could not save this document to the configured database.") from None


def save_study_aids(
    user_id,
    document_id,
    flashcards,
    flashcard_progress,
    quiz,
    quiz_results,
):
    """Save generated flashcards and quiz answers/progress for one document."""
    owner = _user_key(user_id)
    document_id = _document_id(document_id)
    try:
        engine, _, pg_insert, documents, _ = _get_database()
        empty_embeddings = _encode_embeddings(np.empty((0, 0), dtype=np.float32), 0)
        now = datetime.now(timezone.utc)
        statement = pg_insert(documents).values(
            user_id=owner,
            document_id=document_id,
            document_name="",
            document_text="",
            chunks=[],
            embeddings=empty_embeddings,
            flashcards=flashcards,
            flashcard_progress=flashcard_progress,
            quiz=quiz,
            quiz_results=quiz_results,
            updated_at=now,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[documents.c.user_id, documents.c.document_id],
            set_={
                "flashcards": statement.excluded.flashcards,
                "flashcard_progress": statement.excluded.flashcard_progress,
                "quiz": statement.excluded.quiz,
                "quiz_results": statement.excluded.quiz_results,
                "updated_at": statement.excluded.updated_at,
            },
        )
        with engine.begin() as connection:
            connection.execute(statement)
    except StorageError:
        raise
    except Exception:
        raise StorageError(
            "Could not save study progress to the configured database."
        ) from None


def load_user_data(user_id):
    """Load one user's documents, chunks, embedding arrays, and study aids."""
    owner = _user_key(user_id)
    try:
        engine, sa, _, documents, _ = _get_database()
        with engine.connect() as connection:
            rows = connection.execute(
                sa.select(documents).where(documents.c.user_id == owner)
            ).mappings()
            result = {}
            for row in rows:
                chunks = _normalize_chunks(row["chunks"])
                result[row["document_id"]] = {
                    "document_name": row["document_name"],
                    "document_text": row["document_text"],
                    "chunks": chunks,
                    "embeddings": _decode_embeddings(
                        row["embeddings"], len(chunks)
                    ),
                    "flashcards": row["flashcards"],
                    "flashcard_progress": row["flashcard_progress"],
                    "quiz": row["quiz"],
                    "quiz_results": row["quiz_results"],
                }
            return result
    except StorageError:
        raise
    except Exception:
        raise StorageError("Could not load data from the configured database.") from None


def reserve_daily_model_call(user_id, daily_limit=20):
    """Atomically reserve one call for this user on the current UTC date."""
    owner = _user_key(user_id)
    if not isinstance(daily_limit, int) or daily_limit < 1:
        raise ValueError("daily_limit must be a positive integer")
    usage_date = datetime.now(timezone.utc).date()
    try:
        engine, _, pg_insert, _, daily_usage = _get_database()
        statement = pg_insert(daily_usage).values(
            user_id=owner, usage_date=usage_date, call_count=1
        )
        statement = statement.on_conflict_do_update(
            index_elements=[daily_usage.c.user_id, daily_usage.c.usage_date],
            set_={"call_count": daily_usage.c.call_count + 1},
            where=daily_usage.c.call_count < daily_limit,
        )
        with engine.begin() as connection:
            result = connection.execute(statement)
            return result.rowcount == 1
    except StorageError:
        raise
    except Exception:
        raise StorageError("Could not check the daily model-call limit.") from None


def daily_model_calls(user_id):
    """Return the number of calls this user has made today in UTC."""
    owner = _user_key(user_id)
    usage_date = datetime.now(timezone.utc).date()
    try:
        engine, sa, _, _, daily_usage = _get_database()
        with engine.connect() as connection:
            count = connection.execute(
                sa.select(daily_usage.c.call_count).where(
                    daily_usage.c.user_id == owner,
                    daily_usage.c.usage_date == usage_date,
                )
            ).scalar_one_or_none()
        return int(count or 0)
    except StorageError:
        raise
    except Exception:
        raise StorageError("Could not read the daily model-call count.") from None


def delete_user_data(user_id):
    """Delete all saved documents, study aids, and daily usage for one user."""
    owner = _user_key(user_id)
    try:
        engine, sa, _, documents, daily_usage = _get_database()
        with engine.begin() as connection:
            connection.execute(
                sa.delete(documents).where(documents.c.user_id == owner)
            )
            connection.execute(
                sa.delete(daily_usage).where(daily_usage.c.user_id == owner)
            )
    except StorageError:
        raise
    except Exception:
        raise StorageError(
            "Could not delete saved data from the configured database."
        ) from None
