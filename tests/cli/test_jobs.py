"""Tests for the `ts-agents jobs` background-execution commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ts_agents.cli import jobs as jobs_module
from ts_agents.cli import jobs_worker
from ts_agents.cli.main import run


def _wait_for_terminal_status(jobs_root, job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs_module.job_status(jobs_root, job_id)
        if record["status"] in jobs_module.TERMINAL_STATUSES:
            return record
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_jobs_start_runs_command_to_completion(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"

    code = run(
        [
            "jobs",
            "start",
            "--jobs-root",
            str(jobs_root),
            "--json",
            "--",
            "capabilities",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "jobs start"
    job_id = payload["result"]["job_id"]
    assert payload["result"]["argv"] == ["capabilities", "--json"]

    record = _wait_for_terminal_status(jobs_root, job_id)
    assert record["status"] == "completed"
    assert record["exit_code"] == 0
    log_text = (jobs_root / f"{job_id}.log").read_text(encoding="utf-8")
    assert '"command": "capabilities"' in log_text

    code = run(["jobs", "status", job_id, "--jobs-root", str(jobs_root), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "completed"
    assert payload["name"] == job_id

    code = run(
        ["jobs", "logs", job_id, "--jobs-root", str(jobs_root), "--tail", "5", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["result"]["lines"]) == 5


def test_jobs_start_records_failure_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"

    code = run(
        [
            "jobs",
            "start",
            "--jobs-root",
            str(jobs_root),
            "--json",
            "--",
            "runs",
            "show",
            "missing-run",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    job_id = payload["result"]["job_id"]

    record = _wait_for_terminal_status(jobs_root, job_id)
    assert record["status"] == "failed"
    assert record["exit_code"] == 4


def test_jobs_start_requires_a_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = run(["jobs", "start", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"


def test_jobs_start_refuses_unimplemented_windows_supervision(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_module, "_IS_POSIX", False)
    with pytest.raises(jobs_module.ToolError) as exc:
        jobs_module.start_job(["capabilities"], root=tmp_path / "jobs")
    assert exc.value.code == jobs_module.ToolErrorCode.BACKEND_UNAVAILABLE
    assert not (tmp_path / "jobs").exists()


def test_command_group_liveness_without_procfs(monkeypatch):
    process = SimpleNamespace(pid=1234, poll=lambda: 0)
    monkeypatch.setattr(jobs_worker.Path, "glob", lambda *_args: iter(()))
    signals = []
    monkeypatch.setattr(jobs_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert jobs_worker._command_group_alive(process)
    assert signals == [(1234, 0)]


def test_live_command_does_not_scan_procfs(monkeypatch):
    process = SimpleNamespace(pid=1234, poll=lambda: None)
    def fail_scan(*_args):
        raise AssertionError("live leader must not trigger a procfs scan")
    monkeypatch.setattr(jobs_worker.Path, "glob", fail_scan)
    assert jobs_worker._command_group_alive(process)


def test_supervisor_cleans_up_command_when_record_write_fails(tmp_path, monkeypatch):
    import signal

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    original_popen = subprocess.Popen
    monkeypatch.setattr(jobs_worker.subprocess, "Popen", lambda *_args, **_kwargs: process)
    def fail_write(*_args):
        raise OSError("controlled write failure")
    monkeypatch.setattr(jobs_worker, "write_job_record", fail_write)
    try:
        with pytest.raises(OSError, match="controlled write failure"):
            jobs_worker._supervise_command(tmp_path / "record.json", {"argv": ["capabilities"]})
        assert process.poll() is not None
    finally:
        monkeypatch.setattr(jobs_worker.subprocess, "Popen", original_popen)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_cancel_detects_supervisor_loss_during_wait(tmp_path, monkeypatch):
    job_id = "lost-worker"
    jobs_module.write_job_record(jobs_module.job_record_path(tmp_path, job_id), {
        "job_id": job_id, "status": "running", "pid": 1234,
        "cancel_path": str(jobs_module.job_cancel_path(tmp_path, job_id)),
    })
    # Initial effective status and ownership checks pass; the wait then loses it.
    checks = iter([True, True, True, False])
    monkeypatch.setattr(jobs_module, "_worker_identity_matches", lambda _record: next(checks, False))
    result = jobs_module.cancel_job(tmp_path, job_id, wait_seconds=1)
    assert result["status"] == "stale"
    assert "unconfirmed" in result["error"]


def test_jobs_start_rejects_nested_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = run(["jobs", "start", "--json", "--", "jobs", "list"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "validation_error"


def test_jobs_status_unknown_id_returns_not_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = run(["jobs", "status", "missing-job", "--json"])
    assert code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "not_found"


@pytest.mark.skipif(os.name != "posix", reason="real POSIX process-group regression")
@pytest.mark.parametrize("resistant", [False, True])
def test_jobs_cancel_terminates_running_process(tmp_path, monkeypatch, resistant):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()

    job_id = "20260101T000000Z-cancelme"
    record_path = jobs_module.job_record_path(jobs_root, job_id)
    lease_path = jobs_module.job_lease_path(jobs_root, job_id)
    token = "real-supervisor-token"
    jobs_module.initialize_worker_lease(lease_path, token)
    ready = tmp_path / "ready"
    child_pid_path = tmp_path / "child.pid"
    child_script = (
        "import signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if resistant else "")
        + f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    command_script = tmp_path / "command.py"
    command_script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        f"child=subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    jobs_module.write_job_record(record_path, {
        "job_id": job_id, "argv": ["capabilities"], "status": "launching",
        "created_at": jobs_module._utc_now_iso(), "worker_token": token,
        "lease_path": str(lease_path), "pid": None,
        "cancel_path": str(jobs_module.job_cancel_path(jobs_root, job_id)),
    })
    # Exercise the real supervisor/lease/markers with a controlled CLI process
    # that spawns a descendant. Only the command selection is substituted.
    supervisor_script = (
        "import sys\nfrom ts_agents.cli import jobs_worker as w\n"
        "original=w.subprocess.Popen\n"
        f"w.subprocess.Popen=lambda argv, **kw: original([sys.executable, {str(command_script)!r}], **kw)\n"
        f"sys.exit(w.main([{str(record_path)!r}]))\n"
    )
    process = subprocess.Popen([sys.executable, "-c", supervisor_script], start_new_session=True)
    command_pid = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        record = jobs_module.read_job(jobs_root, job_id)
        command_pid = record["command_pid"]
        child_pid = int(child_pid_path.read_text())
        assert jobs_module._worker_identity_matches(record)
        if resistant:
            with pytest.raises(jobs_module.ToolError) as exc:
                jobs_module.cancel_job(jobs_root, job_id, wait_seconds=0.3)
            assert exc.value.code == jobs_module.ToolErrorCode.TIMEOUT
            assert jobs_module.job_status(jobs_root, job_id)["status"] == "running"
            os.kill(child_pid, 0)
        result = jobs_module.cancel_job(jobs_root, job_id, force=resistant)
        assert result["status"] == "cancelled"
        assert result["exit_code"] == 130
        assert process.wait(timeout=10) == 130
        if sys.platform == "linux" and Path(f"/proc/{child_pid}/stat").exists():
            assert Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
        else:
            assert not jobs_module._pid_alive(child_pid)
    finally:
        if command_pid is None:
            command_pid = jobs_module.read_job(jobs_root, job_id).get("command_pid")
        if command_pid is not None:
            try:
                os.killpg(command_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait()


def test_jobs_cancel_stale_record_is_finalized(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-stale001"
    record = {
        "schema_version": jobs_module.JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "argv": ["capabilities"],
        "status": jobs_module.JOB_STATUS_RUNNING,
        "created_at": "2026-01-01T00:00:00Z",
        "pid": 2**22 + 12345,  # almost certainly not a live pid
        "cancel_path": str(jobs_module.job_cancel_path(jobs_root, job_id)),
        "exit_code": None,
        "error": None,
    }
    jobs_module.write_job_record(jobs_module.job_record_path(jobs_root, job_id), record)

    code = run(["jobs", "cancel", job_id, "--jobs-root", str(jobs_root), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "stale"
    assert "unconfirmed" in payload["result"]["error"]


def test_jobs_worker_finalizes_record_in_process(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-worker01"
    record_path = jobs_module.job_record_path(jobs_root, job_id)
    worker_token = "worker-token"
    lease_path = jobs_module.job_lease_path(jobs_root, job_id)
    jobs_module.initialize_worker_lease(lease_path, worker_token)
    jobs_module.write_job_record(
        record_path,
        {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities", "--json"],
            "status": jobs_module.JOB_STATUS_LAUNCHING,
            "created_at": "2026-01-01T00:00:00Z",
            "pid": None,
            "worker_token": worker_token,
            "lease_path": str(lease_path),
            "cancel_path": str(jobs_module.job_cancel_path(jobs_root, job_id)),
            "exit_code": None,
        },
    )

    exit_code = jobs_worker.main([str(record_path)])
    assert exit_code == 0
    final = json.loads(record_path.read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["exit_code"] == 0
    assert final["pid"] == os.getpid()
    assert final["started_at"] is not None
    assert final["finished_at"] is not None


def test_jobs_cancel_during_launch_prevents_command_execution(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-launch01"
    record_path = jobs_module.job_record_path(jobs_root, job_id)
    worker_token = "launch-token"
    lease_path = jobs_module.job_lease_path(jobs_root, job_id)
    jobs_module.initialize_worker_lease(lease_path, worker_token)
    jobs_module.write_job_record(
        record_path,
        {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities", "--json"],
            "status": jobs_module.JOB_STATUS_LAUNCHING,
            "created_at": jobs_module._utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "worker_token": worker_token,
            "lease_path": str(lease_path),
            "cancel_path": str(jobs_module.job_cancel_path(jobs_root, job_id)),
            "exit_code": None,
            "error": None,
        },
    )

    cancelled = jobs_module.cancel_job(jobs_root, job_id)
    assert cancelled["status"] == jobs_module.JOB_STATUS_CANCELLED

    from ts_agents.cli import main as cli_main_module

    executed = False

    def fail_if_executed(_argv):
        nonlocal executed
        executed = True
        raise AssertionError("cancelled command was executed")

    monkeypatch.setattr(cli_main_module, "run", fail_if_executed)
    assert jobs_worker.main([str(record_path)]) == 130
    assert executed is False
    final = json.loads(record_path.read_text(encoding="utf-8"))
    assert final["status"] == jobs_module.JOB_STATUS_CANCELLED


def test_jobs_running_identity_requires_locked_matching_lease(tmp_path):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-lease001"
    lease_path = jobs_module.job_lease_path(jobs_root, job_id)
    worker_token = "identity-token"
    jobs_module.initialize_worker_lease(lease_path, worker_token)
    record = {
        "status": jobs_module.JOB_STATUS_RUNNING,
        "pid": os.getpid(),
        "worker_token": worker_token,
        "lease_path": str(lease_path),
    }

    lease = jobs_module.acquire_worker_lease(lease_path, worker_token)
    try:
        assert jobs_module.effective_status(record) == jobs_module.JOB_STATUS_RUNNING
    finally:
        lease.close()
    assert jobs_module.effective_status(record) == jobs_module.JOB_STATUS_STALE


def test_jobs_cancel_never_signals_unverified_reused_pid(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-reused01"
    lease_path = jobs_module.job_lease_path(jobs_root, job_id)
    jobs_module.initialize_worker_lease(lease_path, "stale-token")
    jobs_module.write_job_record(
        jobs_module.job_record_path(jobs_root, job_id),
        {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities"],
            "status": jobs_module.JOB_STATUS_RUNNING,
            "created_at": jobs_module._utc_now_iso(),
            "pid": os.getpid(),
            "worker_token": "stale-token",
            "lease_path": str(lease_path),
            "cancel_path": str(jobs_module.job_cancel_path(jobs_root, job_id)),
            "exit_code": None,
            "error": None,
        },
    )

    def fail_if_signalled(_pid, *, force=False):
        raise AssertionError(f"unverified pid was signalled (force={force})")

    monkeypatch.setattr(jobs_module, "_terminate_process_tree", fail_if_signalled)
    result = jobs_module.cancel_job(jobs_root, job_id)
    assert result["status"] == jobs_module.JOB_STATUS_STALE
    assert "identity could not be verified" in result["error"]


def test_jobs_logs_tail_zero_is_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-tailzero"
    log_path = jobs_module.job_log_path(jobs_root, job_id)
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    jobs_module.write_job_record(
        jobs_module.job_record_path(jobs_root, job_id),
        {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities"],
            "status": jobs_module.JOB_STATUS_COMPLETED,
            "created_at": jobs_module._utc_now_iso(),
            "pid": None,
            "log_path": str(log_path),
            "exit_code": 0,
        },
    )

    code = run(
        [
            "jobs",
            "logs",
            job_id,
            "--jobs-root",
            str(jobs_root),
            "--tail",
            "0",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["lines"] == []


def test_windows_process_tree_cancellation_uses_group_and_tree(monkeypatch):
    monkeypatch.setattr(jobs_module, "_IS_POSIX", False)
    monkeypatch.setattr(jobs_module, "_IS_WINDOWS", True)
    ctrl_break = getattr(jobs_module.signal, "CTRL_BREAK_EVENT", 21)
    monkeypatch.setattr(
        jobs_module.signal, "CTRL_BREAK_EVENT", ctrl_break, raising=False
    )
    signals = []
    monkeypatch.setattr(
        jobs_module.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )

    jobs_module._terminate_process_tree(1234)
    assert signals == [(1234, ctrl_break)]

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(jobs_module.subprocess, "run", fake_run)
    jobs_module._terminate_process_tree(1234, force=True)
    assert commands == [["taskkill", "/PID", "1234", "/T", "/F"]]


def test_jobs_reject_invalid_numeric_values_and_job_paths(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert run(["jobs", "logs", "job", "--tail", "-1", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage_error"

    assert run(["jobs", "cancel", "job", "--wait", "0", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage_error"

    assert run(["jobs", "cancel", "job", "--wait", "nan", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage_error"

    assert run(["jobs", "status", "../outside", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "validation_error"


def test_jobs_list_reports_all_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    for index, status in enumerate(["completed", "failed"]):
        job_id = f"20260101T00000{index}Z-job{index:05d}"
        jobs_module.write_job_record(
            jobs_module.job_record_path(jobs_root, job_id),
            {
                "schema_version": jobs_module.JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "argv": ["capabilities"],
                "status": status,
                "created_at": f"2026-01-01T00:00:0{index}Z",
                "pid": None,
                "exit_code": 0 if status == "completed" else 6,
            },
        )
    (jobs_root / "not-a-job.json").write_text("[]", encoding="utf-8")

    code = run(["jobs", "list", "--jobs-root", str(jobs_root), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["result"]
    assert [job["status"] for job in result["jobs"]] == ["failed", "completed"]
    assert len(result["warnings"]) == 1
