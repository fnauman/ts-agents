import subprocess
import sys


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
        ["ts-agents-ui", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Time Series Analysis UI" in completed.stdout
