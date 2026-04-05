import json

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
