import filecmp
import json
from pathlib import Path

from ts_agents.cli.main import run


def test_skills_show_json_returns_structured_metadata(capsys):
    code = run(["skills", "show", "forecasting", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "forecasting"
    assert payload["result"]["metadata"]["ts_agents"]["preferred_workflow"] == "forecast-series"
    assert "commands" in payload["result"]
    assert "command_templates" in payload["result"]
    assert payload["result"]["path"].endswith("skills/forecasting/SKILL.md")
    assert any(command.startswith("ts-agents workflow show forecast-series") for command in payload["result"]["commands"])
    assert any(command.startswith("ts-agents workflow run forecast-series") for command in payload["result"]["commands"])
    assert not any(command.endswith("\\") for command in payload["result"]["commands"])


def test_skills_export_json_writes_structured_catalog(tmp_path):
    output_path = tmp_path / "skills_export.json"

    code = run(
        [
            "skills",
            "export",
            "--format",
            "json",
            "--out",
            str(output_path),
        ]
    )

    assert code == 0
    payload = json.loads(output_path.read_text())
    assert "skills" in payload
    skill_names = [skill["name"] for skill in payload["skills"]]
    assert "forecasting" in skill_names
    assert "report-generation" in skill_names


def test_skills_validate_fails_for_invalid_skill(tmp_path, capsys):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken\ndescription: Broken skill for validation.\n---\n\n"
        "# Broken\n\nThis body is long enough but lacks both recommended sections.\n"
    )

    code = run(["skills", "validate", "--path", str(tmp_path), "--json"])

    assert code != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "broken" in payload["error"]["details"]["errors"]


def test_bundled_skills_validate_and_match_packaged_mirror(capsys):
    assert run(["skills", "validate", "--json"]) == 0
    capsys.readouterr()
    root = Path(__file__).resolve().parents[2]
    comparison = filecmp.dircmp(root / "skills", root / "ts_agents" / "resources" / "skills")
    pending = [comparison]
    while pending:
        current = pending.pop()
        assert not (current.left_only or current.right_only or current.diff_files), (
            current.left, current.left_only, current.right_only, current.diff_files,
        )
        pending.extend(current.subdirs.values())


def test_forecasting_skills_mention_every_catalog_method():
    from ts_agents.core.forecasting.catalog import foundation_methods, methods_for

    root = Path(__file__).resolve().parents[2] / "skills"
    panel_skill = (root / "panel-forecasting" / "SKILL.md").read_text()
    forecasting_skill = (root / "forecasting" / "SKILL.md").read_text()

    missing_panel = [name for name in methods_for("panel") if f"`{name}`" not in panel_skill]
    missing_foundation = [
        name for name in foundation_methods() if f"`{name}`" not in forecasting_skill
    ]
    assert not missing_panel, missing_panel
    assert not missing_foundation, missing_foundation
    assert "panel-forecasting" in forecasting_skill
    assert "`foundation`" in panel_skill and "HF_HOME" in panel_skill


def test_panel_skill_frontmatter_and_commands(capsys):
    code = run(["skills", "show", "panel-forecasting", "--json"])

    assert code == 0
    result = json.loads(capsys.readouterr().out)["result"]
    metadata = result["metadata"]
    assert metadata["domain"] == "time-series"
    assert "foundation-models" in metadata["tasks"]
    assert metadata["ts_agents"]["tool_category"] == "forecasting"
    assert metadata["ts_agents"]["preferred_workflow"] == "forecast-panel"
    commands = result["commands"]
    assert any(command.startswith("ts-agents data export-panel m4-monthly-mini") for command in commands)
    assert any("forecast-panel" in command and "chronos2_small" in command for command in commands)


def test_skills_markdown_lists_panel_and_series_workflows():
    from ts_agents.cli.skills import build_skills_markdown

    text = build_skills_markdown()

    assert "- ts-agents workflow run forecast-series " in text
    assert "- ts-agents workflow show forecast-panel --json" in text
    assert "- ts-agents data export-panel m4-monthly-mini" in text
    assert "--methods seasonal_naive,chronos2_small" in text
