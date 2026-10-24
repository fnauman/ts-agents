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
    assert "CLI-first time-series workflows, tools, skills, and sandboxes" in completed.stdout
