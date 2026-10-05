"""Source-bound artifact reports must reject working files from another revision."""

import subprocess

import pytest

from scripts.verify_release_artifacts import verify


def test_artifact_verifier_rejects_dirty_committed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    package = source / "ts_agents"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text('VERSION = "original"\n')
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "ts_agents"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "source"], check=True)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "ts_agents-0.2.0.whl").touch()
    (dist / "ts_agents-0.2.0.tar.gz").touch()
    module.write_text('VERSION = "uncommitted"\n')
    with pytest.raises(ValueError, match="Working source differs from recorded commit"):
        verify(source, dist)
