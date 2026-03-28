import subprocess
import sys

import pytest


def test_python_module_entrypoint_supports_help():
    completed = subprocess.run(
        [sys.executable, "-m", "ts_agents", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Time series analysis CLI" in completed.stdout


def test_ui_console_entrypoint_supports_help():
    completed = subprocess.run(
        [sys.executable, "-m", "ts_agents.ui.entrypoint", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Time Series Analysis UI" in completed.stdout


def test_ui_entrypoint_reports_missing_gradio(monkeypatch, capsys):
    from ts_agents.ui import entrypoint

    def fail_import(name: str):
        raise ModuleNotFoundError("No module named 'gradio'", name="gradio")

    monkeypatch.setattr(entrypoint, "import_module", fail_import)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(["--no-agent"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert 'ts-agents[ui]' in captured.err
