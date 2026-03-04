import argparse
from pathlib import Path

from src.cli.main import _handle_skills_command, build_parser


def test_parser_accepts_skills_export_symlink_flag():
    parser = build_parser()
    args = parser.parse_args(["skills", "export", "--all-agents", "--symlink"])
    assert args.command == "skills"
    assert args.skills_command == "export"
    assert args.all_agents is True
    assert args.symlink is True


def test_handle_skills_export_passes_symlink_flag(monkeypatch):
    import src.cli.skills as cli_skills

    observed = {}

    def fake_export_skills(path, agent=None, all_agents=False, use_symlinks=False):
        observed["path"] = path
        observed["agent"] = agent
        observed["all_agents"] = all_agents
        observed["use_symlinks"] = use_symlinks
        return Path("/tmp/skills-out")

    monkeypatch.setattr(cli_skills, "export_skills", fake_export_skills)

    args = argparse.Namespace(
        skills_command="export",
        out="skills_export",
        agent=None,
        all_agents=True,
        symlink=True,
    )

    result, message = _handle_skills_command(args)

    assert result["skills_path"] == "/tmp/skills-out"
    assert "symlinks" in message
    assert observed == {
        "path": "skills_export",
        "agent": None,
        "all_agents": True,
        "use_symlinks": True,
    }


def _write_minimal_skill(skills_dir: Path, name: str = "diagnostics") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill\n"
        "---\n\n"
        "# Test Skill\n\n"
        "## When to use\n"
        "Use for tests.\n\n"
        "## Workflow\n"
        "1. Run tests.\n"
    )
    return skill_dir


def test_export_skills_agent_respects_output_root(monkeypatch, tmp_path):
    import src.cli.skills as cli_skills

    skills_dir = tmp_path / "skills"
    _write_minimal_skill(skills_dir)
    monkeypatch.setattr(cli_skills, "get_canonical_skills_dir", lambda: skills_dir)

    output_root = tmp_path / "custom-out"
    output_path = cli_skills.export_skills(str(output_root), agent="codex")

    expected_skill_file = output_root / ".codex" / "skills" / "diagnostics" / "SKILL.md"
    assert output_path == output_root / ".codex" / "skills"
    assert expected_skill_file.exists()


def test_export_skills_all_agents_respects_output_root(monkeypatch, tmp_path):
    import src.cli.skills as cli_skills

    skills_dir = tmp_path / "skills"
    _write_minimal_skill(skills_dir)
    monkeypatch.setattr(cli_skills, "get_canonical_skills_dir", lambda: skills_dir)

    output_root = tmp_path / "agents-out"
    output_path = cli_skills.export_skills(str(output_root), all_agents=True)

    assert output_path == output_root
    assert (output_root / ".codex" / "skills" / "diagnostics" / "SKILL.md").exists()
    assert (output_root / ".claude" / "skills" / "time-series-diagnostics" / "SKILL.md").exists()


def test_export_skills_agent_symlink_target_is_valid(monkeypatch, tmp_path):
    import src.cli.skills as cli_skills

    skills_dir = tmp_path / "skills"
    source_skill = _write_minimal_skill(skills_dir)
    source_file = source_skill / "SKILL.md"
    monkeypatch.setattr(cli_skills, "get_canonical_skills_dir", lambda: skills_dir)

    output_root = tmp_path / "symlink-out"
    cli_skills.export_skills(str(output_root), agent="codex", use_symlinks=True)

    target = output_root / ".codex" / "skills" / "diagnostics" / "SKILL.md"
    assert target.is_symlink()
    assert target.resolve() == source_file.resolve()


def test_get_canonical_skills_dir_uses_runtime_default(monkeypatch, tmp_path):
    import src.cli.skills as cli_skills
    from src.runtime_paths import resolve_default_skills_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TS_AGENTS_SKILLS_DIR", raising=False)

    assert cli_skills.get_canonical_skills_dir() == resolve_default_skills_dir()


def test_build_skills_markdown_uses_cli_script_examples():
    import src.cli.skills as cli_skills

    markdown = cli_skills.build_skills_markdown()
    assert "ts-agents data list" in markdown
    assert "python -m ts_agents ...  # supported compatibility entrypoint" in markdown
