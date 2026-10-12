"""Launch with `python -m app.worker` in a separate process from Uvicorn.

A single transactional reservation bounds queued/running work across API
processes. Conditional claim prevents two worker processes training one job.
Only persisted training statistics are emitted; no simulated training events.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timezone

from sqlalchemy import select, update

from . import db

logger = logging.getLogger(__name__)
_STOP = False


def recover_stale_jobs():
    timeout = max(120, int(os.getenv("TRAINING_STALE_SECONDS", "900")))
    with db.session_scope() as session:
        slot = session.get(db.TrainingSlot, 1)
        if not slot or not slot.job_id:
            return
        job = session.get(db.TrainingJob, slot.job_id)
        if job is None:
            slot.job_id = None
            return
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(job.updated_at)).total_seconds()
        if job.status == "running" and age > timeout:
            job.status = "failed"
            job.error = "Worker stopped reporting progress. Submit a new bounded training job."
            job.updated_at = db.utcnow()
            slot.job_id = None
            db.emit(session, "training", db.job_dict(job))


def claim_job():
    with db.session_scope() as session:
        slot = session.get(db.TrainingSlot, 1)
        if not slot or not slot.job_id:
            return None
        claimed = session.execute(update(db.TrainingJob).where(
            db.TrainingJob.id == slot.job_id, db.TrainingJob.status == "queued"
        ).values(status="running", updated_at=db.utcnow()))
        if claimed.rowcount != 1:
            return None
        session.expire_all()
        job = session.get(db.TrainingJob, slot.job_id)
        db.emit(session, "training", db.job_dict(job))
        return db.job_dict(job)


def run_next_job():
    job = claim_job()
    if not job:
        return False
    job_id = job["id"]

    def capture_progress(progress):
        with db.session_scope() as session:
            row = session.get(db.TrainingJob, job_id)
            if row.status != "running":
                raise RuntimeError("Training job no longer owns its active reservation")
            payload = {**progress, "epochs": job["config"]["epochs"]}
            row.progress = payload
            row.history = [*row.history, payload][-20:]
            row.updated_at = db.utcnow()
            db.emit(session, "training", db.job_dict(row))

    try:
        from .ml import train_model
        metadata = train_model(**job["config"], progress_callback=capture_progress)
        from .main import register_models
        with db.session_scope() as session:
            row = session.get(db.TrainingJob, job_id)
            if row.status != "running":
                raise RuntimeError("Training reservation expired before checkpoint registration")
            register_models(session)
            row.model_id = metadata.get("id", metadata.get("model_id"))
            row.status = "completed"
            row.updated_at = db.utcnow()
            slot = session.get(db.TrainingSlot, 1)
            if slot.job_id == job_id:
                slot.job_id = None
            db.emit(session, "training", db.job_dict(row))
    except BaseException as error:
        logger.exception("Training job %s failed", job_id)
        with db.session_scope() as session:
            row = session.get(db.TrainingJob, job_id)
            row.status = "failed"
            # Training errors contain no provider credentials. Keep messages bounded.
            row.error = f"{type(error).__name__}: {str(error)[:500]}"
            row.updated_at = db.utcnow()
            slot = session.get(db.TrainingSlot, 1)
            if slot.job_id == job_id:
                slot.job_id = None
            db.emit(session, "training", db.job_dict(row))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    return True


def request_stop(signum, _frame):
    global _STOP
    _STOP = True
    if signum == signal.SIGTERM:
        raise SystemExit("Worker stopped")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    from .main import seed_database, maintain_audit
    seed_database()
    logger.info("Training worker started; one bounded training job at a time")
    poll = max(0.2, min(float(os.getenv("WORKER_POLL_SECONDS", "1")), 30))
    maintenance_at = 0.0
    while not _STOP:
        try:
            if time.monotonic() - maintenance_at > 30:
                recover_stale_jobs()
                maintain_audit()
                maintenance_at = time.monotonic()
            if not run_next_job():
                time.sleep(poll)
        except Exception:
            logger.exception("Worker polling failed; will retry")
            time.sleep(min(poll * 5, 30))


if __name__ == "__main__":
    main()
