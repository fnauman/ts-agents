"""Tests for the `ts-agents jobs` background-execution commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

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
        ["jobs", "start", "--jobs-root", str(jobs_root), "--json", "--", "capabilities", "--json"]
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

    code = run(["jobs", "logs", job_id, "--jobs-root", str(jobs_root), "--tail", "5", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["result"]["lines"]) == 5


def test_jobs_start_records_failure_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"

    code = run(["jobs", "start", "--jobs-root", str(jobs_root), "--json", "--", "runs", "show", "missing-run"])
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


def test_jobs_cancel_terminates_running_process(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()

    popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        **popen_kwargs,
    )
    try:
        job_id = "20260101T000000Z-cancelme"
        record = {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities"],
            "status": jobs_module.JOB_STATUS_RUNNING,
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": None,
            "pid": process.pid,
            "exit_code": None,
            "error": None,
            "log_path": str(jobs_root / f"{job_id}.log"),
        }
        jobs_module.write_job_record(jobs_module.job_record_path(jobs_root, job_id), record)

        code = run(["jobs", "cancel", job_id, "--jobs-root", str(jobs_root), "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"]["status"] == "cancelled"
        assert process.wait(timeout=10) is not None
    finally:
        if process.poll() is None:
            process.kill()


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
        "exit_code": None,
        "error": None,
    }
    jobs_module.write_job_record(jobs_module.job_record_path(jobs_root, job_id), record)

    code = run(["jobs", "cancel", job_id, "--jobs-root", str(jobs_root), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "cancelled"


def test_jobs_worker_finalizes_record_in_process(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    job_id = "20260101T000000Z-worker01"
    record_path = jobs_module.job_record_path(jobs_root, job_id)
    jobs_module.write_job_record(
        record_path,
        {
            "schema_version": jobs_module.JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "argv": ["capabilities", "--json"],
            "status": jobs_module.JOB_STATUS_LAUNCHING,
            "created_at": "2026-01-01T00:00:00Z",
            "pid": None,
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
