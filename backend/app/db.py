"""Persistent audit records shared by API and the separately launched worker.

Forecasts, input snapshots, telemetry, and outcomes are append-only. PostgreSQL
triggers enforce that guarantee; SQLite receives matching triggers for local use.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'neurostock.db'}")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Instrument(Base):
    __tablename__ = "instruments"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    exchange: Mapped[str] = mapped_column(String(32), default="NASDAQ")
    timezone: Mapped[str] = mapped_column(String(64), default="America/New_York")


class DataSource(Base):
    __tablename__ = "data_sources"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(32), default="Replay")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Observation(Base):
    __tablename__ = "market_observations"
    __table_args__ = (UniqueConstraint("symbol", "time", "source_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("instruments.symbol"), index=True)
    time: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("data_sources.id"))
    values: Mapped[dict] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    metadata_json: Mapped[dict] = mapped_column(JSON)


class PreprocessingArtifact(Base):
    __tablename__ = "preprocessing_artifacts"
    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), unique=True)
    metadata_json: Mapped[dict] = mapped_column(JSON)


class DataSnapshot(Base):
    __tablename__ = "data_snapshots"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16))
    input_cutoff: Mapped[str] = mapped_column(String(32))
    digest: Mapped[str] = mapped_column(String(64))
    candles: Mapped[list] = mapped_column(JSON)
    source_id: Mapped[str] = mapped_column(ForeignKey("data_sources.id"))
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    issue_time: Mapped[str] = mapped_column(String(40), index=True)
    target_time: Mapped[str] = mapped_column(String(32), index=True)
    input_cutoff: Mapped[str] = mapped_column(String(32))
    data_snapshot_id: Mapped[str] = mapped_column(ForeignKey("data_snapshots.id"))
    payload: Mapped[dict] = mapped_column(JSON)


class ForecastOutcome(Base):
    __tablename__ = "forecast_outcomes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.id"), unique=True)
    observed_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    target_time: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


class TelemetrySnapshot(Base):
    __tablename__ = "telemetry_snapshots"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.id"), unique=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Watchlist(Base):
    __tablename__ = "watchlists"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default="local")
    symbols: Mapped[list] = mapped_column(JSON, default=list)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    config: Mapped[dict] = mapped_column(JSON)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    history: Mapped[list] = mapped_column(JSON, default=list)
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    updated_at: Mapped[str] = mapped_column(String(40), default=utcnow)


class TrainingSlot(Base):
    """One row is a transactional reservation, shared across API processes."""
    __tablename__ = "training_slots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), unique=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow)
    payload: Mapped[dict] = mapped_column(JSON)


class StreamEvent(Base):
    __tablename__ = "stream_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(24), index=True)
    timestamp: Mapped[str] = mapped_column(String(40), default=utcnow)
    payload: Mapped[dict] = mapped_column(JSON)


IMMUTABLE_TABLES = ("forecasts", "forecast_outcomes", "data_snapshots", "telemetry_snapshots", "model_versions", "preprocessing_artifacts", "evaluation_runs")


def build_engine(url: str):
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if url.endswith(":memory:"):
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def install_immutability(connection):
    dialect = connection.dialect.name
    if dialect == "sqlite":
        for table in IMMUTABLE_TABLES:
            for operation in ("UPDATE", "DELETE"):
                connection.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable audit record'); END"
                )
    elif dialect == "postgresql":
        connection.exec_driver_sql("""
            CREATE OR REPLACE FUNCTION neurostock_reject_audit_mutation() RETURNS trigger AS $$
            BEGIN RAISE EXCEPTION 'immutable audit record'; END; $$ LANGUAGE plpgsql
        """)
        for table in IMMUTABLE_TABLES:
            connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS immutable_{table} ON {table}")
            connection.exec_driver_sql(
                f"CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION neurostock_reject_audit_mutation()"
            )


def init_db():
    # PostgreSQL schemas are deployed with Alembic, SQLite bootstraps locally.
    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            install_immutability(connection)
    with session_scope() as session:
        insert_ignore(session, TrainingSlot, {"id": 1, "job_id": None})


def insert_ignore(session, model, values):
    """Atomically seed shared rows even if API and worker start together."""
    if session.bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return session.execute(insert(model).values(**values).on_conflict_do_nothing())


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def emit(session, kind: str, payload: dict):
    if session.bind.dialect.name == "postgresql":
        # Sequence ids alone do not imply commit order. Serialize event writers
        # until commit so a reconnect cursor cannot skip a late committing event.
        session.execute(text("SELECT pg_advisory_xact_lock(81263942)"))
    event_row = StreamEvent(type=kind, payload=payload, timestamp=utcnow())
    session.add(event_row)
    session.flush()
    return event_row


def event_dict(row: StreamEvent):
    return {"id": str(row.id), "type": row.type, "timestamp": row.timestamp, "payload": row.payload}


def job_dict(row: TrainingJob):
    return {"id": row.id, "status": row.status, "config": row.config,
            "progress": row.progress, "history": row.history, "model_id": row.model_id,
            "error": row.error, "created_at": row.created_at, "updated_at": row.updated_at}
