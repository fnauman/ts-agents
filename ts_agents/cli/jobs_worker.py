"""Detached worker process behind ``ts-agents jobs start``.

Invoked as ``python -m ts_agents.cli.jobs_worker <record-path>``. Reads the
job record written by :func:`ts_agents.cli.jobs.start_job`, stamps itself as
running, re-runs the CLI with the recorded argv, and finalizes the record
with the exit code. Stdout/stderr are already redirected to the job log by
the launcher.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from typing import List, Optional

from ts_agents.cli.jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    TERMINAL_STATUSES,
    _cancellation_requested,
    _utc_now_iso,
    acquire_worker_lease,
    write_job_record,
)


_CANCELLED_EXIT_CODE = 130


def _finalize_cancelled(record_path: Path, record: dict) -> int:
    """Finalize a cancellation without ever entering the requested command."""
    try:
        current = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(current, dict):
            record = current
    except (OSError, json.JSONDecodeError):
        pass
    record["status"] = JOB_STATUS_CANCELLED
    record["finished_at"] = record.get("finished_at") or _utc_now_iso()
    record["exit_code"] = (
        record.get("exit_code")
        if record.get("exit_code") is not None
        else _CANCELLED_EXIT_CODE
    )
    write_job_record(record_path, record)
    return _CANCELLED_EXIT_CODE


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m ts_agents.cli.jobs_worker <job-record-path>",
            file=sys.stderr,
        )
        return 2

    record_path = Path(args[0])
    record = json.loads(record_path.read_text(encoding="utf-8"))

    import os

    worker_token = record.get("worker_token")
    lease_path = record.get("lease_path")
    if not isinstance(worker_token, str) or not isinstance(lease_path, str):
        raise RuntimeError("Job record is missing its worker ownership lease.")
    lease = acquire_worker_lease(lease_path, worker_token)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") in TERMINAL_STATUSES:
            if record.get("status") == JOB_STATUS_CANCELLED:
                return _CANCELLED_EXIT_CODE
            existing_exit = record.get("exit_code")
            return int(existing_exit) if isinstance(existing_exit, int) else 0
        if _cancellation_requested(record):
            return _finalize_cancelled(record_path, record)

        # The lease is locked before this transition. Any later status/cancel
        # command can therefore verify that this PID still owns this exact job.
        record["status"] = JOB_STATUS_RUNNING
        record["pid"] = os.getpid()
        record["started_at"] = _utc_now_iso()
        write_job_record(record_path, record)

        # Cancellation publishes its marker before reading or writing the
        # record. Checking again after the running transition closes the
        # launch-time race without requiring the launcher to poll for a PID.
        if _cancellation_requested(record):
            return _finalize_cancelled(record_path, record)

        exit_code = 6
        error: Optional[str] = None
        try:
            from ts_agents.cli.main import run as cli_run

            exit_code = int(cli_run(record["argv"]))
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 6
        except BaseException as exc:  # noqa: BLE001 - the record must always finalize
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            exit_code = 6

        # Re-read before finalizing: `jobs cancel` may have stamped the record
        # while we were finishing, and cancellation must win.
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if _cancellation_requested(record):
            record["status"] = JOB_STATUS_CANCELLED
        elif record.get("status") not in TERMINAL_STATUSES:
            record["status"] = (
                JOB_STATUS_COMPLETED if exit_code == 0 else JOB_STATUS_FAILED
            )
        record["exit_code"] = exit_code
        record["finished_at"] = _utc_now_iso()
        if error is not None:
            record["error"] = error
        write_job_record(record_path, record)
        return exit_code
    finally:
        lease.close()


if __name__ == "__main__":
    sys.exit(main())
