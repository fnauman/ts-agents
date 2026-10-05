"""Verify wheel/sdist source identity and wheel RECORD integrity before upload."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tomllib
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(source: Path, dist: Path) -> dict:
    wheels = list(dist.glob("ts_agents-*.whl"))
    sdists = list(dist.glob("ts_agents-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", "-z", commit], text=True,
    ).split("\0")
    package_files = [name for name in tracked if name.startswith("ts_agents/")]
    root_files = {"AGENTS.md", "CHANGELOG.md", "ROADMAP.md", "LICENSE", "README.md", "pyproject.toml",
                  "uv.lock", "main.py", "app.py", "MANIFEST.in"}
    maintained_prefixes = ("ts_agents/", "scripts/", "tests/", "skills/", "examples/", "data/")
    source_files = [name for name in tracked if name in root_files or name.startswith(maintained_prefixes)]
    if not package_files:
        raise ValueError("Source checkout has no tracked package files")
    git_archive = subprocess.check_output(["git", "-C", str(source), "archive", commit])
    with tarfile.open(fileobj=io.BytesIO(git_archive)) as archive:
        committed_files = {}
        for name in source_files:
            extracted = archive.extractfile(name)
            if extracted is None:
                raise ValueError(f"Missing committed source file: {name}")
            committed_files[name] = extracted.read()
            if (source / name).read_bytes() != committed_files[name]:
                raise ValueError(f"Working source differs from recorded commit: {name}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        project = tomllib.loads(committed_files["pyproject.toml"].decode())["project"]
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_path).decode())
        if metadata["Version"] != project["version"] or metadata["Requires-Python"] != project["requires-python"]:
            raise ValueError("Wheel version/Python requirement differs from source metadata")
        actual_package = {name for name in wheel.namelist() if name.startswith("ts_agents/") and not name.endswith("/")}
        if actual_package != set(package_files):
            raise ValueError(f"Wheel package inventory mismatch: {sorted(actual_package ^ set(package_files))}")
        for name in package_files:
            if wheel.read(name) != committed_files[name]:
                raise ValueError(f"Wheel source mismatch: {name}")
        record_names = [name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("Expected one wheel RECORD")
        rows = list(csv.reader(io.StringIO(wheel.read(record_names[0]).decode())))
        if len(rows) != len({row[0] for row in rows}):
            raise ValueError("Duplicate wheel RECORD entries")
        if {row[0] for row in rows} != {name for name in wheel.namelist() if not name.endswith("/")}:
            raise ValueError("RECORD inventory mismatch")
        for name, digest, size in rows:
            data = wheel.read(name)
            if name == record_names[0]:
                if digest or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if digest != expected or size != str(len(data)):
                raise ValueError(f"RECORD integrity mismatch: {name}")
    with tarfile.open(sdists[0]) as archive:
        members = archive.getmembers()
        for member in members:
            name = member.name.removesuffix("/") if member.isdir() else member.name
            if (PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
                    or ".." in PurePosixPath(name).parts or "\\" in name):
                raise ValueError(f"Noncanonical source archive member: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Nonregular source archive member: {member.name}")
        names = [member.name.removesuffix("/") if member.isdir() else member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source archive member paths")
        files = {member.name for member in members if member.isfile()}
        directories = {name for name, member in zip(names, members) if member.isdir()}
        for name in names:
            directories.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
        if files & directories:
            raise ValueError(f"Source archive file/directory collision: {sorted(files & directories)}")
        roots = {member.name.split("/")[0] for member in members}
        if len(roots) != 1:
            raise ValueError("Expected one source archive root")
        root = roots.pop()
        if root != sdists[0].name.removesuffix(".tar.gz"):
            raise ValueError("Source archive root differs from distribution filename")
        generated_files = {"PKG-INFO", "setup.cfg", *(
            f"ts_agents.egg-info/{name}" for name in (
                "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"))}
        unexpected = {name.removeprefix(f"{root}/") for name in files} - set(source_files) - generated_files
        if unexpected:
            raise ValueError(f"Unexpected source archive members: {sorted(unexpected)}")
        maintained_members = [member for member in members if not member.isdir()
                              and (member.name.removeprefix(f"{root}/") in root_files
                                   or member.name.removeprefix(f"{root}/").startswith(maintained_prefixes))]
        actual_source = {member.name.removeprefix(f"{root}/") for member in maintained_members}
        if actual_source != set(source_files):
            raise ValueError(f"Source archive inventory mismatch: {sorted(actual_source ^ set(source_files))}")
        for name in source_files:
            extracted = archive.extractfile(f"{root}/{name}")
            if extracted is None or extracted.read() != committed_files[name]:
                raise ValueError(f"Source archive mismatch: {name}")
    source_hashes = {name: sha256(committed_files[name]) for name in sorted(source_files)}
    return {
        "version": project["version"],
        "commit": commit,
        "tree": subprocess.check_output(["git", "-C", str(source), "rev-parse", f"{commit}^{{tree}}"], text=True).strip(),
        "source_digest": sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()),
        "source_files": source_hashes,
        "wheel_package_files_verified": len(package_files),
        "sdist_source_files_verified": len(source_files),
        "wheel_record_entries_verified": len(rows),
        "artifacts": {path.name: sha256(path.read_bytes()) for path in (wheels[0], sdists[0])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.source_root.resolve(), args.dist_dir.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {result['wheel_package_files_verified']} wheel source files, "
          f"{result['sdist_source_files_verified']} sdist files, and wheel RECORD")


if __name__ == "__main__":
    main()
