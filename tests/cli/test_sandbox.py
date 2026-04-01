import json

from ts_agents.cli.main import run
from ts_agents.tools.executor import ExecutionResult, ExecutionStatus, ToolError, ToolErrorCode


def test_sandbox_list_json_returns_envelope(capsys):
    code = run(["sandbox", "list", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "sandbox list"
    backend_names = [backend["backend"] for backend in payload["result"]["backends"]]
    assert "local" in backend_names
    assert "docker" in backend_names


def test_sandbox_doctor_local_json_returns_backend_status(capsys):
    code = run(["sandbox", "doctor", "local", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "local"
    assert payload["result"]["backend"] == "local"
    assert payload["result"]["available"] is True


def test_tool_run_backend_unavailable_returns_typed_json(monkeypatch, capsys):
    import ts_agents.tools.executor as executor_mod

    def fake_execute_tool(tool_name, params, context=None):
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            error=ToolError(
                code=ToolErrorCode.BACKEND_UNAVAILABLE,
                message="Requested backend 'docker' is unavailable and fallback is not allowed.",
                recoverable=True,
                tool_name=tool_name,
            ),
            metadata={
                "backend_requested": "docker",
                "backend_actual": None,
                "fallback_used": False,
            },
        )

    monkeypatch.setattr(executor_mod, "execute_tool", fake_execute_tool)

    code = run(
        [
            "tool",
            "run",
            "describe_series",
            "--param",
            "series=[1,2,3]",
            "--sandbox",
            "docker",
            "--json",
        ]
    )

    assert code == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "backend_unavailable"
    assert payload["execution"]["backend_requested"] == "docker"
