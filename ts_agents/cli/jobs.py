"""Background job control plane for long-running CLI invocations.

``ts-agents jobs start -- <ts-agents args...>`` launches a detached worker
process (``ts_agents.cli.jobs_worker``) that re-runs the CLI with the given
arguments, streams combined stdout/stderr to a log file, and finalizes a JSON
job record with the exit code. The record is the single source of truth:
``status``/``cancel``/``logs`` only ever read or signal, so any later CLI
invocation (or another agent) can manage the job.
"""

from __future__ import annotations

from datetime import datetime, timezone
import errno
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from ts_agents.tools.executor import ToolError, ToolErrorCode
from ts_agents.workflows.common import generate_workflow_run_id

DEFAULT_JOBS_ROOT = "outputs/jobs"
JOB_SCHEMA_VERSION = "1.0"

JOB_STATUS_LAUNCHING = "launching"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_STALE = "stale"

TERMINAL_STATUSES = {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}

# A job that never left "launching" after this long is presumed dead.
_LAUNCH_GRACE_SECONDS = 120.0
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_IS_POSIX = os.name == "posix"
_IS_WINDOWS = os.name == "nt"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_job_id(job_id: str) -> str:
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError(
            "job_id must contain only letters, numbers, '.', '_', and '-' "
            "and must not contain path separators."
        )
    return job_id


def job_record_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / f"{_validate_job_id(job_id)}.json"


def job_log_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / f"{_validate_job_id(job_id)}.log"


def job_lease_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / f"{_validate_job_id(job_id)}.lease"


def job_cancel_path(root: str | Path, job_id: str) -> Path:
    return Path(root) / f"{_validate_job_id(job_id)}.cancel"


def write_job_record(path: str | Path, record: Dict[str, Any]) -> None:
    """Write the record atomically so concurrent readers never see partial JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        tmp_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_job(root: str | Path, job_id: str) -> Dict[str, Any]:
    record_path = job_record_path(root, job_id)
    if not record_path.is_file():
        raise ToolError(
            code=ToolErrorCode.NOT_FOUND,
            message=f"No job with id {job_id!r} under {Path(root).resolve()}.",
            hint="Use `ts-agents jobs list` to see known jobs.",
        )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ToolError(
            code=ToolErrorCode.DATA_ERROR,
            message=f"Job record {record_path} does not contain a JSON object.",
        )
    return payload


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _try_lock_lease(handle: Any) -> bool:
    """Try to acquire the worker lease without blocking."""
    if _IS_POSIX:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        try:
            getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True
    raise RuntimeError("Background job leases are unsupported on this platform.")


def _unlock_lease(handle: Any) -> None:
    if _IS_POSIX:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)


def initialize_worker_lease(path: str | Path, token: str) -> None:
    """Create the ownership-lease file before the detached worker is spawned."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{token}\n", encoding="utf-8")


def acquire_worker_lease(path: str | Path, token: str) -> Any:
    """Lock and return a worker lease, proving ownership until it is closed."""
    handle = open(path, "r+b")
    try:
        stored_token = handle.read().decode("utf-8").strip()
        if not secrets.compare_digest(stored_token, token):
            raise RuntimeError("Worker lease token does not match the job record.")
        if not _try_lock_lease(handle):
            raise RuntimeError("Worker lease is already owned by another process.")
    except BaseException:
        handle.close()
        raise
    return handle


def _worker_identity_matches(record: Dict[str, Any]) -> bool:
    """Verify that the recorded PID still owns this job's locked lease."""
    pid = record.get("pid")
    token = record.get("worker_token")
    lease_path = record.get("lease_path")
    if (
        not isinstance(pid, int)
        or not isinstance(token, str)
        or not token
        or not isinstance(lease_path, str)
        or not _pid_alive(pid)
    ):
        return False
    try:
        with open(lease_path, "r+b") as handle:
            stored_token = handle.read().decode("utf-8").strip()
            if not secrets.compare_digest(stored_token, token):
                return False
            if not _try_lock_lease(handle):
                return True
            _unlock_lease(handle)
            return False
    except (OSError, RuntimeError, UnicodeDecodeError):
        return False


def _cancellation_requested(record: Dict[str, Any]) -> bool:
    cancel_path = record.get("cancel_path")
    return isinstance(cancel_path, str) and Path(cancel_path).is_file()


def _request_cancellation(record: Dict[str, Any]) -> None:
    cancel_path = record.get("cancel_path")
    if not isinstance(cancel_path, str):
        raise ToolError(
            code=ToolErrorCode.DATA_ERROR,
            message="Job record has no cancellation-marker path.",
        )
    marker = Path(cancel_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def effective_status(record: Dict[str, Any]) -> str:
    """Resolve the live status, detecting workers that died without finalizing."""
    status = record.get("status")
    if status in TERMINAL_STATUSES:
        return str(status)
    if status == JOB_STATUS_RUNNING:
        return (
            JOB_STATUS_RUNNING if _worker_identity_matches(record) else JOB_STATUS_STALE
        )
    if status == JOB_STATUS_LAUNCHING:
        created_at = record.get("created_at")
        try:
            created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return JOB_STATUS_STALE
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if age > _LAUNCH_GRACE_SECONDS and not _pid_alive(record.get("pid")):
            return JOB_STATUS_STALE
        return JOB_STATUS_LAUNCHING
    return str(status)


def _record_view(record: Dict[str, Any]) -> Dict[str, Any]:
    view = dict(record)
    view["status"] = effective_status(record)
    return view


def start_job(
    argv: List[str],
    *,
    root: str | Path = DEFAULT_JOBS_ROOT,
) -> Dict[str, Any]:
    """Launch ``ts-agents <argv...>`` in a detached background worker."""
    if not argv:
        raise ValueError(
            "jobs start requires a command to run, e.g. "
            "`ts-agents jobs start -- workflow run forecast ...`"
        )
    if argv[0] == "jobs":
        raise ValueError("jobs start cannot launch nested `jobs` commands.")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    job_id = generate_workflow_run_id()
    record_path = job_record_path(root_path, job_id)
    log_path = job_log_path(root_path, job_id)
    lease_path = job_lease_path(root_path, job_id)
    cancel_path = job_cancel_path(root_path, job_id)
    worker_token = secrets.token_hex(32)

    record: Dict[str, Any] = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "argv": list(argv),
        "status": JOB_STATUS_LAUNCHING,
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "worker_token": worker_token,
        "lease_path": str(lease_path.resolve()),
        "cancel_path": str(cancel_path.resolve()),
        "exit_code": None,
        "error": None,
        "cwd": os.getcwd(),
        "record_path": str(record_path.resolve()),
        "log_path": str(log_path.resolve()),
    }
    # The record must exist before the worker starts: the worker reads its
    # argv from it and stamps pid/status itself to avoid racing this process.
    initialize_worker_lease(lease_path, worker_token)
    write_job_record(record_path, record)

    popen_kwargs: Dict[str, Any] = {}
    if _IS_POSIX:
        popen_kwargs["start_new_session"] = True
    elif _IS_WINDOWS and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    with open(log_path, "ab") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "ts_agents.cli.jobs_worker", str(record_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **popen_kwargs,
        )

    view = dict(record)
    view["pid"] = process.pid
    return view


def list_jobs(root: str | Path = DEFAULT_JOBS_ROOT) -> Dict[str, Any]:
    root_path = Path(root)
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if root_path.exists():
        for record_path in sorted(root_path.glob("*.json")):
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"Skipped unreadable job record {record_path}: {exc}")
                continue
            if not isinstance(payload, dict) or "job_id" not in payload:
                warnings.append(f"Skipped non-job JSON file {record_path}")
                continue
            records.append(_record_view(payload))
    records.sort(key=lambda record: str(record.get("created_at")), reverse=True)
    return {"root": str(root_path.resolve()), "jobs": records, "warnings": warnings}


def job_status(root: str | Path, job_id: str) -> Dict[str, Any]:
    return _record_view(read_job(root, job_id))


def read_job_log(
    root: str | Path,
    job_id: str,
    *,
    tail: Optional[int] = None,
) -> Dict[str, Any]:
    record = read_job(root, job_id)
    # Derive the log path from the validated id rather than trusting a mutable
    # path stored inside the JSON record.
    log_path = job_log_path(root, job_id)
    if not log_path.is_file():
        content = ""
    else:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    if tail is not None:
        if tail < 0:
            raise ValueError("--tail must be zero or a positive integer.")
        lines = lines[-tail:] if tail else []
    return {
        "job_id": job_id,
        "status": effective_status(record),
        "log_path": str(log_path),
        "lines": lines,
    }


def _terminate_process_tree(pid: int, *, force: bool = False) -> None:
    sig = signal.SIGKILL if force and hasattr(signal, "SIGKILL") else signal.SIGTERM
    if _IS_POSIX:
        try:
            process_group = os.getpgid(pid)
            if process_group != pid:
                raise ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message=(
                        f"Refusing to signal process group {process_group}: "
                        f"job worker {pid} is not its leader."
                    ),
                    recoverable=True,
                )
            os.killpg(process_group, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise ToolError(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Permission denied while signalling job worker {pid}.",
                recoverable=True,
            ) from exc
        return
    if _IS_WINDOWS:
        if not force and hasattr(signal, "CTRL_BREAK_EVENT"):
            try:
                os.kill(pid, signal.CTRL_BREAK_EVENT)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 or not _pid_alive(pid):
            return
        detail = (completed.stderr or completed.stdout).strip()
        raise ToolError(
            code=ToolErrorCode.PERMISSION_DENIED,
            message=f"Could not terminate the process tree for job worker {pid}.",
            recoverable=True,
            details={"taskkill_output": detail} if detail else {},
        )
    raise ToolError(
        code=ToolErrorCode.BACKEND_UNAVAILABLE,
        message="Background job cancellation is unsupported on this platform.",
    )


def cancel_job(
    root: str | Path,
    job_id: str,
    *,
    force: bool = False,
    wait_seconds: float = 10.0,
) -> Dict[str, Any]:
    """Signal a running job's process group and stamp the record cancelled."""
    if not math.isfinite(wait_seconds) or wait_seconds <= 0:
        raise ValueError("--wait must be a finite positive number of seconds.")
    record = read_job(root, job_id)
    status = effective_status(record)
    if status in TERMINAL_STATUSES:
        return _record_view(record)
    _request_cancellation(record)
    # Re-read after publishing the marker. A worker that has not started its
    # command yet is required to observe this marker before doing so.
    record = read_job(root, job_id)
    status = effective_status(record)
    if status in TERMINAL_STATUSES:
        return _record_view(record)
    pid = record.get("pid")
    if status == JOB_STATUS_LAUNCHING:
        record["status"] = JOB_STATUS_CANCELLED
        record["finished_at"] = _utc_now_iso()
        record["error"] = record.get("error") or "cancelled before worker startup"
        write_job_record(job_record_path(root, job_id), record)
        return _record_view(record)
    if (
        status == JOB_STATUS_STALE
        or not isinstance(pid, int)
        or not _worker_identity_matches(record)
    ):
        record["status"] = JOB_STATUS_CANCELLED
        record["finished_at"] = _utc_now_iso()
        record["error"] = (
            record.get("error")
            or "worker identity could not be verified before cancellation"
        )
        write_job_record(job_record_path(root, job_id), record)
        return _record_view(record)

    _terminate_process_tree(pid, force=force)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        # If the job process happens to be our child, reap it so it does not
        # linger as a zombie that os.kill(pid, 0) still reports as alive.
        if hasattr(os, "waitpid") and hasattr(os, "WNOHANG"):
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
                if reaped == pid:
                    break
            except (ChildProcessError, OSError):
                pass
        if not _worker_identity_matches(record):
            break
        time.sleep(0.1)
    else:
        raise ToolError(
            code=ToolErrorCode.TIMEOUT,
            message=(
                f"Job {job_id} (pid {pid}) did not exit within "
                f"{wait_seconds:.0f}s of SIGTERM."
            ),
            recoverable=True,
            hint="Retry with --force to send SIGKILL.",
        )

    # Re-read in case the worker finalized between the signal and now.
    record = read_job(root, job_id)
    if record.get("status") not in TERMINAL_STATUSES:
        record["status"] = JOB_STATUS_CANCELLED
        record["finished_at"] = _utc_now_iso()
        write_job_record(job_record_path(root, job_id), record)
    return _record_view(record)


def render_jobs_table(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "No jobs found."
    headers = ("JOB_ID", "STATUS", "PID", "EXIT", "CREATED_AT", "COMMAND")
    rows = []
    for record in records:
        argv = record.get("argv") or []
        command = " ".join(str(part) for part in argv)
        if len(command) > 48:
            command = command[:45] + "..."
        rows.append(
            (
                str(record.get("job_id") or "-"),
                str(record.get("status") or "-"),
                str(record.get("pid") or "-"),
                str(
                    record.get("exit_code")
                    if record.get("exit_code") is not None
                    else "-"
                ),
                str(record.get("created_at") or "-"),
                command or "-",
            )
        )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
