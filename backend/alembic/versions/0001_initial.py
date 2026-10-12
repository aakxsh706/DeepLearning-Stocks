"""Initial persistent research/audit schema and immutable forecast safeguards.

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # This release's schema is frozen here; future schema changes get new revisions.
    op.create_table("instruments", sa.Column("symbol", sa.String(16), primary_key=True),
                    sa.Column("name", sa.String(128), nullable=False), sa.Column("exchange", sa.String(32), nullable=False),
                    sa.Column("timezone", sa.String(64), nullable=False))
    op.create_table("data_sources", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("name", sa.String(128), nullable=False), sa.Column("mode", sa.String(32), nullable=False),
                    sa.Column("metadata_json", sa.JSON(), nullable=False))
    op.create_table("market_observations", sa.Column("id", sa.Integer(), primary_key=True),
                    sa.Column("symbol", sa.String(16), sa.ForeignKey("instruments.symbol"), nullable=False),
                    sa.Column("time", sa.String(32), nullable=False),
                    sa.Column("source_id", sa.String(64), sa.ForeignKey("data_sources.id"), nullable=False),
                    sa.Column("values", sa.JSON(), nullable=False), sa.UniqueConstraint("symbol", "time", "source_id"))
    op.create_index("ix_market_observations_symbol", "market_observations", ["symbol"])
    op.create_index("ix_market_observations_time", "market_observations", ["time"])
    op.create_table("model_versions", sa.Column("id", sa.String(128), primary_key=True),
                    sa.Column("symbol", sa.String(16), nullable=False), sa.Column("kind", sa.String(24), nullable=False),
                    sa.Column("created_at", sa.String(40), nullable=False), sa.Column("metadata_json", sa.JSON(), nullable=False))
    op.create_index("ix_model_versions_symbol", "model_versions", ["symbol"])
    op.create_table("preprocessing_artifacts", sa.Column("id", sa.String(160), primary_key=True),
                    sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.id"), nullable=False, unique=True),
                    sa.Column("metadata_json", sa.JSON(), nullable=False))
    op.create_table("data_snapshots", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("symbol", sa.String(16), nullable=False), sa.Column("input_cutoff", sa.String(32), nullable=False),
                    sa.Column("digest", sa.String(64), nullable=False), sa.Column("candles", sa.JSON(), nullable=False),
                    sa.Column("source_id", sa.String(64), sa.ForeignKey("data_sources.id"), nullable=False),
                    sa.Column("created_at", sa.String(40), nullable=False))
    op.create_table("forecasts", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("symbol", sa.String(16), nullable=False),
                    sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.id"), nullable=False),
                    sa.Column("issue_time", sa.String(40), nullable=False), sa.Column("target_time", sa.String(32), nullable=False),
                    sa.Column("input_cutoff", sa.String(32), nullable=False),
                    sa.Column("data_snapshot_id", sa.String(64), sa.ForeignKey("data_snapshots.id"), nullable=False),
                    sa.Column("payload", sa.JSON(), nullable=False))
    for column in ("symbol", "model_id", "issue_time", "target_time"):
        op.create_index(f"ix_forecasts_{column}", "forecasts", [column])
    op.create_table("forecast_outcomes", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("forecast_id", sa.String(64), sa.ForeignKey("forecasts.id"), nullable=False, unique=True),
                    sa.Column("observed_at", sa.String(40), nullable=False), sa.Column("target_time", sa.String(32), nullable=False),
                    sa.Column("payload", sa.JSON(), nullable=False))
    op.create_table("telemetry_snapshots", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("forecast_id", sa.String(64), sa.ForeignKey("forecasts.id"), nullable=False, unique=True),
                    sa.Column("payload", sa.JSON(), nullable=False))
    op.create_table("watchlists", sa.Column("id", sa.String(64), primary_key=True), sa.Column("symbols", sa.JSON(), nullable=False))
    op.create_table("training_jobs", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("status", sa.String(24), nullable=False), sa.Column("config", sa.JSON(), nullable=False),
                    sa.Column("progress", sa.JSON(), nullable=True), sa.Column("history", sa.JSON(), nullable=False),
                    sa.Column("model_id", sa.String(128), nullable=True), sa.Column("error", sa.Text(), nullable=True),
                    sa.Column("created_at", sa.String(40), nullable=False), sa.Column("updated_at", sa.String(40), nullable=False))
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])
    op.create_table("training_slots", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("job_id", sa.String(64), nullable=True))
    op.create_table("evaluation_runs", sa.Column("id", sa.String(64), primary_key=True),
                    sa.Column("model_id", sa.String(128), sa.ForeignKey("model_versions.id"), unique=True, nullable=False),
                    sa.Column("created_at", sa.String(40), nullable=False), sa.Column("payload", sa.JSON(), nullable=False))
    op.create_table("stream_events", sa.Column("id", sa.Integer(), primary_key=True),
                    sa.Column("type", sa.String(24), nullable=False), sa.Column("timestamp", sa.String(40), nullable=False),
                    sa.Column("payload", sa.JSON(), nullable=False))
    op.create_index("ix_stream_events_type", "stream_events", ["type"])
    tables = ("forecasts", "forecast_outcomes", "data_snapshots", "telemetry_snapshots", "model_versions", "preprocessing_artifacts", "evaluation_runs")
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        op.execute("CREATE FUNCTION neurostock_reject_audit_mutation() RETURNS trigger AS $$ BEGIN RAISE EXCEPTION 'immutable audit record'; END; $$ LANGUAGE plpgsql")
        for table in tables:
            op.execute(f"CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION neurostock_reject_audit_mutation()")
    elif connection.dialect.name == "sqlite":
        for table in tables:
            for operation in ("UPDATE", "DELETE"):
                op.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable audit record'); END")


def downgrade():
    for table in ("stream_events", "evaluation_runs", "training_slots", "training_jobs", "watchlists", "telemetry_snapshots",
                  "forecast_outcomes", "forecasts", "data_snapshots", "preprocessing_artifacts", "model_versions", "market_observations", "data_sources", "instruments"):
        op.drop_table(table)
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP FUNCTION IF EXISTS neurostock_reject_audit_mutation()")
