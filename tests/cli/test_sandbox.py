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


def test_capabilities_json_returns_bootstrap_surface(capsys):
    code = run(["capabilities", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "capabilities"
    assert "entrypoints" in payload["result"]
    assert "status_contract" in payload["result"]
    assert "install_profile" in payload["result"]
    assert "workflows" in payload["result"]
    assert "tools" in payload["result"]
    assert "sandboxes" in payload["result"]
    install_profile = payload["result"]["install_profile"]
    assert install_profile["package"] == "ts-agents"
    assert "current_profile" in install_profile
    assert install_profile["recommended_install"] == "ts-agents[recommended]"
    assert "forecasting" in install_profile["extras"]


def test_sandbox_doctor_local_json_returns_backend_status(capsys):
    code = run(["sandbox", "doctor", "local", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "local"
    assert payload["result"]["backend"] == "local"
    assert payload["result"]["available"] is True


def test_sandbox_doctor_docker_json_reports_missing_image(monkeypatch, capsys):
    def fake_describe(_backend, context=None, backend=None):
        return {
            "backend": "docker",
            "description": "Containerized execution through Docker.",
            "available": False,
            "reason": "Docker image 'ts-agents-sandbox:latest' is not available locally.",
            "suggested_fix": "Build or pull 'ts-agents-sandbox:latest', set TS_AGENTS_DOCKER_IMAGE to an available image, or run with --sandbox local.",
            "requirements": [
                "Docker CLI installed.",
                "Docker daemon running.",
                "Sandbox image available.",
            ],
            "details": {"image": "ts-agents-sandbox:latest"},
        }

    import ts_agents.tools.executor as executor_mod
    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)

    code = run(["sandbox", "doctor", "docker", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "docker"
    assert payload["result"]["available"] is False
    assert payload["result"]["reason"] == "Docker image 'ts-agents-sandbox:latest' is not available locally."


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
