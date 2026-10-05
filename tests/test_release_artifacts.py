"""Source-bound artifact reports must reject working files from another revision."""

import base64
import csv
import hashlib
import io
import subprocess
import tarfile
import zipfile

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


@pytest.mark.parametrize("extra", ["tests/extra.py", "scripts/extra.py", "skills/extra.md", "ts_agents/__init__.py",
                                   "./ts_agents/__init__.py", "ts_agents/../ts_agents/__init__.py",
                                   "ts_agents\\__init__.py", "alias-link", "ts_agents", "metadata-collision", "setup.py"])
def test_artifact_verifier_rejects_unrecorded_or_unsafe_sdist_members(tmp_path, extra):
    source = tmp_path / "source"
    source.mkdir()
    package = source / "ts_agents"
    package.mkdir()
    (package / "__init__.py").write_text("# recorded source\n")
    (source / "pyproject.toml").write_text('[project]\nversion = "0.2.1"\nrequires-python = ">=3.11"\n')
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "ts_agents", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "source"], check=True)
    dist = tmp_path / "dist"
    dist.mkdir()
    files = {"ts_agents/__init__.py": (package / "__init__.py").read_bytes(),
             "ts_agents-0.2.1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: ts-agents\nVersion: 0.2.1\nRequires-Python: >=3.11\n",
             "ts_agents-0.2.1.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = io.StringIO()
    writer = csv.writer(record)
    for name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        writer.writerow([name, "sha256=" + digest, str(len(content))])
    writer.writerow(["ts_agents-0.2.1.dist-info/RECORD", "", ""])
    files["ts_agents-0.2.1.dist-info/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(dist / "ts_agents-0.2.1-py3-none-any.whl", "w") as wheel:
        for name, content in files.items():
            wheel.writestr(name, content)
    archive_path = dist / "ts_agents-0.2.1.tar.gz"
    def write_sdist(include_extra):
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in ["ts_agents/__init__.py", "pyproject.toml"]:
                archive.add(source / name, arcname=f"ts_agents-0.2.1/{name}")
            if include_extra:
                if extra == "metadata-collision":
                    leaf = tarfile.TarInfo("ts_agents-0.2.1/metadata-collision/leaf.txt")
                    archive.addfile(leaf, io.BytesIO(b""))
                content = b"# content outside the recorded inventory\n"
                member = tarfile.TarInfo(f"ts_agents-0.2.1/{extra}")
                if extra == "alias-link":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "ts_agents"
                    archive.addfile(member)
                else:
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
    write_sdist(False)
    assert verify(source, dist)["sdist_source_files_verified"] == 2
    write_sdist(True)
    message = ("file/directory collision" if extra in {"ts_agents", "metadata-collision"}
               else "inventory mismatch|Duplicate|Noncanonical source|Nonregular source|Unexpected source")
    with pytest.raises(ValueError, match=message):
        verify(source, dist)
