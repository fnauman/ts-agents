"""Tests for the `ts-agents runs` control-plane commands."""

from __future__ import annotations

import json
from pathlib import Path

from ts_agents.cli.main import run


def _write_workflow_manifest(
    output_dir: Path,
    *,
    run_id: str,
    workflow: str = "inspect-series",
    status: str = "ok",
    created_at: str = "2026-05-01T12:00:00Z",
    artifacts: list | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "workflow": workflow,
        "run_id": run_id,
        "status": status,
        "summary": f"{workflow} run {run_id}",
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "run_manifest.json"),
        "created_at": created_at,
        "artifacts": artifacts or [],
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_autoresearch_manifest(
    output_dir: Path,
    *,
    run_id: str,
    loop: str = "forecast-daytona",
    status: str = "ok",
    created_at: str = "2026-05-02T12:00:00Z",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "kind": "autoresearch_run",
        "loop": loop,
        "run_id": run_id,
        "status": status,
        "summary": f"{loop} run {run_id}",
        "created_at": created_at,
        "artifacts": [],
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_runs_list_empty_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = run(["runs", "list", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "runs list"
    assert payload["result"]["runs"] == []
    assert payload["result"]["count"] == 0


def test_runs_list_normalizes_both_manifest_shapes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    _write_workflow_manifest(
        outputs / "inspect" / "20260501T120000Z-aaaa1111",
        run_id="20260501T120000Z-aaaa1111",
        created_at="2026-05-01T12:00:00Z",
    )
    _write_autoresearch_manifest(
        outputs / "autoresearch" / "forecast-daytona" / "20260502T120000Z-bbbb2222",
        run_id="20260502T120000Z-bbbb2222",
        created_at="2026-05-02T12:00:00Z",
    )
    # Legacy flat layout: manifest one level below the root.
    _write_workflow_manifest(
        outputs / "demo",
        run_id="20260430T120000Z-cccc3333",
        workflow="demo",
        created_at="2026-04-30T12:00:00Z",
    )
    (outputs / "broken").mkdir(parents=True)
    (outputs / "broken" / "run_manifest.json").write_text("{not json", encoding="utf-8")

    code = run(["runs", "list", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    runs = payload["result"]["runs"]
    assert [entry["run_id"] for entry in runs] == [
        "20260502T120000Z-bbbb2222",
        "20260501T120000Z-aaaa1111",
        "20260430T120000Z-cccc3333",
    ]
    kinds = {entry["run_id"]: entry["kind"] for entry in runs}
    assert kinds["20260502T120000Z-bbbb2222"] == "autoresearch"
    assert kinds["20260501T120000Z-aaaa1111"] == "workflow"
    names = {entry["run_id"]: entry["name"] for entry in runs}
    assert names["20260502T120000Z-bbbb2222"] == "forecast-daytona"
    assert names["20260501T120000Z-aaaa1111"] == "inspect-series"
    assert len(payload["result"]["warnings"]) == 1
    assert "broken" in payload["result"]["warnings"][0]


def test_runs_list_filters_and_limit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    _write_workflow_manifest(
        outputs / "inspect" / "run-a",
        run_id="run-a",
        status="ok",
        created_at="2026-05-01T12:00:00Z",
    )
    _write_workflow_manifest(
        outputs / "inspect" / "run-b",
        run_id="run-b",
        status="degraded",
        created_at="2026-05-02T12:00:00Z",
    )
    _write_autoresearch_manifest(
        outputs / "autoresearch" / "loop" / "run-c",
        run_id="run-c",
        created_at="2026-05-03T12:00:00Z",
    )

    code = run(["runs", "list", "--kind", "workflow", "--status", "degraded", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["run_id"] for entry in payload["result"]["runs"]] == ["run-b"]

    code = run(["runs", "list", "--limit", "1", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["run_id"] for entry in payload["result"]["runs"]] == ["run-c"]


def test_runs_show_resolves_unique_prefix_and_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs" / "inspect" / "20260501T120000Z-aaaa1111"
    _write_workflow_manifest(
        output_dir,
        run_id="20260501T120000Z-aaaa1111",
        artifacts=[
            {"kind": "report", "path": str(output_dir / "report.md")},
            {"kind": "json", "path": str(output_dir / "missing.json")},
        ],
    )
    (output_dir / "report.md").write_text("# report", encoding="utf-8")

    code = run(["runs", "show", "20260501T120000Z", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["result"]
    assert result["run_id"] == "20260501T120000Z-aaaa1111"
    assert result["size_bytes"] > 0
    exists = {entry["path"]: entry["exists"] for entry in result["artifacts"]}
    assert exists[str(output_dir / "report.md")] is True
    assert exists[str(output_dir / "missing.json")] is False


def test_runs_show_unknown_id_returns_not_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = run(["runs", "show", "nope", "--json"])
    assert code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_found"


def test_runs_show_ambiguous_prefix_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    _write_workflow_manifest(outputs / "inspect" / "run-a1", run_id="run-a1")
    _write_workflow_manifest(outputs / "inspect" / "run-a2", run_id="run-a2")

    code = run(["runs", "show", "run-a", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "validation_error"


def test_runs_gc_dry_run_then_apply(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    old_dir = outputs / "inspect" / "old-run"
    new_dir = outputs / "inspect" / "new-run"
    _write_workflow_manifest(old_dir, run_id="old-run", created_at="2020-01-01T00:00:00Z")
    _write_workflow_manifest(new_dir, run_id="new-run", created_at="2099-01-01T00:00:00Z")

    code = run(["runs", "gc", "--older-than", "30", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["dry_run"] is True
    assert [entry["run_id"] for entry in payload["result"]["runs"]] == ["old-run"]
    assert old_dir.exists()

    code = run(["runs", "gc", "--older-than", "30", "--apply", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["dry_run"] is False
    assert not old_dir.exists()
    assert new_dir.exists()


def test_runs_gc_refuses_root_and_escaping_directories(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    # A manifest directly in the outputs root must never delete the root.
    _write_workflow_manifest(outputs, run_id="root-run", created_at="2020-01-01T00:00:00Z")

    code = run(["runs", "gc", "--older-than", "1", "--apply", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["runs"] == []
    skipped = payload["result"]["skipped"]
    assert len(skipped) == 1
    assert skipped[0]["run_id"] == "root-run"
    assert outputs.exists()
