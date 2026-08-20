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
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    TERMINAL_STATUSES,
    _utc_now_iso,
    write_job_record,
)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m ts_agents.cli.jobs_worker <job-record-path>", file=sys.stderr)
        return 2

    record_path = Path(args[0])
    record = json.loads(record_path.read_text(encoding="utf-8"))

    import os

    record["status"] = JOB_STATUS_RUNNING
    record["pid"] = os.getpid()
    record["started_at"] = _utc_now_iso()
    write_job_record(record_path, record)

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
    if record.get("status") not in TERMINAL_STATUSES:
        record["status"] = JOB_STATUS_COMPLETED if exit_code == 0 else JOB_STATUS_FAILED
    record["exit_code"] = exit_code
    record["finished_at"] = _utc_now_iso()
    if error is not None:
        record["error"] = error
    write_job_record(record_path, record)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
