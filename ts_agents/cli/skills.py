"""Skills export, validation, and multi-agent placement helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ts_agents.runtime_paths import resolve_default_skills_dir


# Skill directory mappings for different agents
AGENT_SKILL_PATHS = {
    "claude": ".claude/skills",
    "codex": ".codex/skills",
    "gemini": ".gemini/skills",
    "windsurf": ".windsurf/skills",
    "github": ".github/skills",
}

# Required frontmatter fields for validation
REQUIRED_FRONTMATTER_FIELDS = ["name", "description"]
OPTIONAL_FRONTMATTER_FIELDS = ["compatibility", "metadata"]

# Agent-specific metadata fields
CLAUDE_CODE_FIELDS = ["disable-model-invocation", "allowed-tools"]


def _format_tools_by_category() -> List[str]:
    """Format tools organized by category for SKILLS.md."""
    from ts_agents.tools.bundles import CATEGORY_BUNDLES

    lines: List[str] = []
    for category_name, tools in CATEGORY_BUNDLES.items():
        if not tools:
            continue
        lines.append(f"### {category_name}")
        for tool_name in tools:
            lines.append(f"- {tool_name}")
        lines.append("")

    return lines


def build_skills_markdown() -> str:
    """Build the aggregate SKILLS.md summary file."""
    from ts_agents.tools.bundles import (
        CATEGORY_BUNDLES,
        DEMO_BUNDLE,
        DEMO_WINDOWING_BUNDLE,
        DEMO_FORECASTING_BUNDLE,
        MINIMAL_BUNDLE,
        STANDARD_BUNDLE,
        FULL_BUNDLE,
        ORCHESTRATOR_BUNDLE,
    )

    lines: List[str] = [
        "# SKILLS.md",
        "",
        "Auto-generated skills summary for ts-agents.",
        "",
        "## CLI",
        "- ts-agents data list",
        "- ts-agents tool list",
        "- ts-agents tool show forecast_theta_with_data",
        "- ts-agents tool run <tool> --run <RUN> --var <VAR>",
        "- ts-agents workflow run inspect-series --input-json '{\"series\":[1,2,3,4]}'",
        "- ts-agents workflow run activity-recognition --input stream.csv --label-col label --value-cols x,y,z",
        "- ts-agents sandbox list",
        "- ts-agents skills show forecasting",
        "- ts-agents agent run \"<prompt>\"",
        "- ts-agents skills export --out skills_export",
        "- ts-agents skills export --format json --out skills_export.json",
        "- ts-agents skills validate",
        "- python -m ts_agents ...  # supported module entrypoint",
        "",
        "## Tool Bundles",
    ]

    bundle_summaries = {
        "demo": f"Meta-bundle covering both demo tracks ({len(DEMO_BUNDLE)} tools)",
        "demo_windowing": (
            f"Focused windowing essentials for LLM-first demo ({len(DEMO_WINDOWING_BUNDLE)} tools)"
        ),
        "demo_forecasting": f"Focused forecasting demo tools ({len(DEMO_FORECASTING_BUNDLE)} tools)",
        "minimal": f"Core tools for baseline testing ({len(MINIMAL_BUNDLE)} tools)",
        "standard": f"Well-rounded set for typical analysis ({len(STANDARD_BUNDLE)} tools)",
        "full": f"Comprehensive set for in-depth analysis ({len(FULL_BUNDLE)} tools)",
        "orchestrator": "High-level tools for agent orchestration",
    }
    for name, description in bundle_summaries.items():
        lines.append(f"- {name}: {description}")
    for category_name, tools in CATEGORY_BUNDLES.items():
        lines.append(f"- {category_name}: All {category_name} tools ({len(tools)} tools)")

    lines.append("")
    lines.append("## Tools by Category")
    lines.append("")
    lines.extend(_format_tools_by_category())

    return "\n".join(lines).strip() + "\n"


def get_canonical_skills_dir() -> Path:
    """Get the canonical skills directory path."""
    env_override = os.environ.get("TS_AGENTS_SKILLS_DIR")
    if env_override:
        return Path(env_override).expanduser()

    # Try to find project root
    current = Path.cwd()
    while current != current.parent:
        if (current / "skills").is_dir():
            return current / "skills"
        current = current.parent

    return resolve_default_skills_dir()


def list_skills(skills_dir: Optional[Path] = None) -> List[Path]:
    """List all skill directories in the given or canonical directory.

    Parameters
    ----------
    skills_dir : Path, optional
        Directory to scan. Defaults to canonical skills directory.

    Returns
    -------
    List[Path]
        List of paths to skill directories containing SKILL.md
    """
    skills_dir = skills_dir or get_canonical_skills_dir()

    if not skills_dir.exists():
        return []

    skill_dirs = []
    for item in skills_dir.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skill_dirs.append(item)

    return sorted(skill_dirs)


def parse_skill_frontmatter(skill_path: Path) -> Tuple[Dict, str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Parameters
    ----------
    skill_path : Path
        Path to the SKILL.md file

    Returns
    -------
    Tuple[Dict, str]
        (frontmatter dict, body content)
    """
    content = skill_path.read_text()

    # Parse YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return frontmatter, body
            except yaml.YAMLError:
                pass

    return {}, content


def _extract_markdown_sections(body: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped[level:].strip()
        if not title:
            continue
        sections.append(
            {
                "level": level,
                "title": title,
                "slug": title.lower().replace(" ", "-"),
            }
        )
    return sections


def _normalize_ts_agents_command(command: str) -> str:
    normalized = " ".join(command.strip().split())
    if normalized.startswith("uv run ts-agents "):
        return "ts-agents " + normalized[len("uv run ts-agents ") :]
    return normalized


def _extract_command_sequences(lines: List[str]) -> List[str]:
    commands: List[str] = []
    current: List[str] = []

    def _maybe_finalize() -> None:
        if not current:
            return
        commands.append(_normalize_ts_agents_command(" ".join(current)))
        current.clear()

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            _maybe_finalize()
            continue

        is_start = stripped.startswith("uv run ts-agents ") or stripped.startswith("ts-agents ")
        if not current and not is_start:
            continue

        continuation = stripped.endswith("\\")
        chunk = stripped[:-1].rstrip() if continuation else stripped
        current.append(chunk)

        if not continuation:
            _maybe_finalize()

    _maybe_finalize()
    return commands


def _extract_ts_agents_commands(body: str) -> List[str]:
    commands: List[str] = []
    in_fence = False
    fence_lines: List[str] = []
    prose_lines: List[str] = []

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_fence:
                commands.extend(_extract_command_sequences(fence_lines))
                fence_lines = []
                in_fence = False
            else:
                in_fence = True
            continue

        if in_fence:
            fence_lines.append(raw_line)
        else:
            prose_lines.append(raw_line)

    if fence_lines:
        commands.extend(_extract_command_sequences(fence_lines))

    commands.extend(_extract_command_sequences(prose_lines))

    unique_commands: List[str] = []
    for command in commands:
        if command not in unique_commands:
            unique_commands.append(command)
    return unique_commands


def get_skill_details(
    skill_name: str,
    skills_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return structured metadata for a single skill."""
    skills_dir = skills_dir or get_canonical_skills_dir()
    skill_dir = skills_dir / skill_name
    skill_file = skill_dir / "SKILL.md"

    if not skill_dir.exists():
        raise ValueError(f"Skill '{skill_name}' not found in {skills_dir}")
    if not skill_file.exists():
        raise ValueError(f"SKILL.md not found in {skill_dir}")

    frontmatter, body = parse_skill_frontmatter(skill_file)
    commands = _extract_ts_agents_commands(body)
    return {
        "name": skill_dir.name,
        "path": str(skill_file),
        "description": frontmatter.get("description", ""),
        "compatibility": frontmatter.get("compatibility"),
        "metadata": frontmatter.get("metadata", {}),
        "frontmatter": frontmatter,
        "sections": _extract_markdown_sections(body),
        "commands": commands,
        "command_templates": commands,
        "body": body,
    }


def build_skills_catalog(skills_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build a machine-readable catalog for all skills."""
    skills_dir = skills_dir or get_canonical_skills_dir()
    return {
        "skills_dir": str(skills_dir),
        "skills": [get_skill_details(skill_dir.name, skills_dir=skills_dir) for skill_dir in list_skills(skills_dir)],
    }


def validate_skill(skill_dir: Path) -> List[str]:
    """Validate a single skill directory.

    Parameters
    ----------
    skill_dir : Path
        Path to the skill directory

    Returns
    -------
    List[str]
        List of validation errors (empty if valid)
    """
    errors: List[str] = []
    skill_file = skill_dir / "SKILL.md"

    # Check SKILL.md exists
    if not skill_file.exists():
        errors.append(f"Missing SKILL.md in {skill_dir.name}")
        return errors

    # Parse frontmatter
    frontmatter, body = parse_skill_frontmatter(skill_file)

    # Check required fields
    for field in REQUIRED_FRONTMATTER_FIELDS:
        if field not in frontmatter:
            errors.append(f"{skill_dir.name}: Missing required field '{field}'")

    # Validate name matches directory
    if "name" in frontmatter:
        expected_name = skill_dir.name
        actual_name = frontmatter["name"]
        # Allow time-series- prefix variations
        normalized_expected = expected_name.replace("time-series-", "")
        normalized_actual = actual_name.replace("time-series-", "")
        # This is just a warning, not an error

    # Check metadata structure
    if "metadata" in frontmatter:
        metadata = frontmatter["metadata"]
        if not isinstance(metadata, dict):
            errors.append(f"{skill_dir.name}: 'metadata' must be a dictionary")
        else:
            # Check ts_agents section
            if "ts_agents" in metadata:
                ts_meta = metadata["ts_agents"]
                if not isinstance(ts_meta, dict):
                    errors.append(f"{skill_dir.name}: 'ts_agents' metadata must be a dictionary")
                # Validate tool_category vs tool_categories naming
                has_category = "tool_category" in ts_meta
                has_categories = "tool_categories" in ts_meta
                if has_category and has_categories:
                    errors.append(
                        f"{skill_dir.name}: Use either 'tool_category' or 'tool_categories', not both"
                    )
                preferred_workflow = ts_meta.get("preferred_workflow")
                if preferred_workflow is not None and not isinstance(preferred_workflow, str):
                    errors.append(f"{skill_dir.name}: 'preferred_workflow' must be a string")
                min_series_length = ts_meta.get("min_series_length")
                if min_series_length is not None and (
                    not isinstance(min_series_length, int) or min_series_length < 1
                ):
                    errors.append(f"{skill_dir.name}: 'min_series_length' must be a positive integer")
                for list_field in ("preferred_tools", "avoid_tools", "artifact_checklist"):
                    value = ts_meta.get(list_field)
                    if value is not None and (
                        not isinstance(value, list) or any(not isinstance(item, str) for item in value)
                    ):
                        errors.append(
                            f"{skill_dir.name}: '{list_field}' must be a list of strings"
                        )

    # Check body has content
    if len(body.strip()) < 50:
        errors.append(f"{skill_dir.name}: Body content seems too short")

    # Check for required sections
    required_sections = ["when to use", "workflow"]
    body_lower = body.lower()
    for section in required_sections:
        if section not in body_lower and f"## {section}" not in body_lower:
            # Allow variations like "What this skill does" for "when to use"
            if section == "when to use" and "what this skill" in body_lower:
                continue
            errors.append(f"{skill_dir.name}: Missing recommended section '{section}'")

    return errors


def validate_all_skills(skills_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """Validate all skills in a directory.

    Parameters
    ----------
    skills_dir : Path, optional
        Directory to validate. Defaults to canonical skills directory.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of skill name to list of errors (empty dict if all valid)
    """
    skills_dir = skills_dir or get_canonical_skills_dir()
    skill_dirs = list_skills(skills_dir)

    all_errors: Dict[str, List[str]] = {}
    for skill_dir in skill_dirs:
        errors = validate_skill(skill_dir)
        if errors:
            all_errors[skill_dir.name] = errors

    return all_errors


def place_skills_for_agent(
    agent: str,
    skills_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    use_symlinks: bool = False,
) -> List[Path]:
    """Place skills in an agent-specific directory.

    Parameters
    ----------
    agent : str
        Agent name (claude, codex, gemini, windsurf, github)
    skills_dir : Path, optional
        Source skills directory
    project_root : Path, optional
        Project root for agent-specific paths
    use_symlinks : bool
        Whether to use symlinks instead of copies

    Returns
    -------
    List[Path]
        Paths to placed skill files
    """
    if agent not in AGENT_SKILL_PATHS:
        raise ValueError(f"Unknown agent '{agent}'. Available: {list(AGENT_SKILL_PATHS.keys())}")

    skills_dir = skills_dir or get_canonical_skills_dir()
    project_root = project_root or skills_dir.parent

    agent_path = project_root / AGENT_SKILL_PATHS[agent]
    skill_dirs = list_skills(skills_dir)

    placed_files: List[Path] = []

    for skill_dir in skill_dirs:
        skill_name = skill_dir.name
        source_file = skill_dir / "SKILL.md"

        # Determine target directory name based on agent conventions
        if agent == "claude":
            # Claude uses longer names with time-series- prefix
            target_dir_name = f"time-series-{skill_name}" if not skill_name.startswith("time-series-") else skill_name
            # Special case for tool-authoring
            if skill_name == "tool-authoring":
                target_dir_name = "ts-agents-tool-authoring"
        else:
            # Other agents use the short canonical name
            target_dir_name = skill_name

        target_dir = agent_path / target_dir_name
        target_file = target_dir / "SKILL.md"

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing file/symlink
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()

        if use_symlinks:
            # Create relative symlink
            relative_source = Path(os.path.relpath(source_file, start=target_dir))
            target_file.symlink_to(relative_source)
        else:
            # Copy file
            shutil.copy2(source_file, target_file)

        placed_files.append(target_file)

    return placed_files


def place_skills_all_agents(
    skills_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    use_symlinks: bool = False,
) -> Dict[str, List[Path]]:
    """Place skills for all supported agents.

    Parameters
    ----------
    skills_dir : Path, optional
        Source skills directory
    project_root : Path, optional
        Project root for agent-specific paths
    use_symlinks : bool
        Whether to use symlinks instead of copies

    Returns
    -------
    Dict[str, List[Path]]
        Mapping of agent name to list of placed files
    """
    result: Dict[str, List[Path]] = {}

    for agent in AGENT_SKILL_PATHS.keys():
        try:
            placed = place_skills_for_agent(
                agent=agent,
                skills_dir=skills_dir,
                project_root=project_root,
                use_symlinks=use_symlinks,
            )
            result[agent] = placed
        except Exception as e:
            result[agent] = []  # Log error but continue

    return result


def export_individual_skill(
    skill_name: str,
    output_dir: Path,
    skills_dir: Optional[Path] = None,
) -> Path:
    """Export a single skill to a directory.

    Parameters
    ----------
    skill_name : str
        Name of the skill to export
    output_dir : Path
        Output directory
    skills_dir : Path, optional
        Source skills directory

    Returns
    -------
    Path
        Path to the exported SKILL.md file
    """
    skills_dir = skills_dir or get_canonical_skills_dir()
    skill_dir = skills_dir / skill_name

    if not skill_dir.exists():
        raise ValueError(f"Skill '{skill_name}' not found in {skills_dir}")

    source_file = skill_dir / "SKILL.md"
    if not source_file.exists():
        raise ValueError(f"SKILL.md not found in {skill_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_dir / skill_name / "SKILL.md"
    target_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_file, target_file)
    return target_file


def export_skills(
    path: str,
    agent: Optional[str] = None,
    all_agents: bool = False,
    use_symlinks: bool = False,
    format_name: Optional[str] = None,
) -> Path:
    """Export skills to a path.

    Parameters
    ----------
    path : str
        Output path (file or directory)
    agent : str, optional
        Specific agent to export for
    all_agents : bool
        Export to all agent-specific locations
    use_symlinks : bool
        Whether to create symlinks for agent exports instead of copies

    Returns
    -------
    Path
        Output path
    """
    output_path = Path(path)
    effective_format = format_name or ("json" if output_path.suffix.lower() == ".json" else "markdown")

    if all_agents:
        if effective_format == "json":
            raise ValueError("--format json is not supported with --all-agents")
        # Place skills for all agents under the requested output root.
        if output_path.exists() and output_path.is_file():
            raise ValueError(f"Output path must be a directory for --all-agents: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        skills_dir = get_canonical_skills_dir()
        place_skills_all_agents(
            skills_dir=skills_dir,
            project_root=output_path,
            use_symlinks=use_symlinks,
        )
        return output_path

    if agent:
        if effective_format == "json":
            raise ValueError("--format json is not supported with --agent")
        # Place skills for specific agent under the requested output root.
        if output_path.exists() and output_path.is_file():
            raise ValueError(f"Output path must be a directory for --agent: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        skills_dir = get_canonical_skills_dir()
        place_skills_for_agent(
            agent=agent,
            skills_dir=skills_dir,
            project_root=output_path,
            use_symlinks=use_symlinks,
        )
        return output_path / AGENT_SKILL_PATHS[agent]

    if effective_format == "json":
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / "skills_export.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(build_skills_catalog(), indent=2))
        return output_path

    # Default: export aggregate SKILLS.md
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / "SKILLS.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_skills_markdown())
    return output_path


def add_agent_metadata(
    skill_path: Path,
    agent: str,
    metadata: Dict[str, str],
) -> None:
    """Add agent-specific metadata fields to a skill.

    Parameters
    ----------
    skill_path : Path
        Path to the SKILL.md file
    agent : str
        Agent name (claude, codex, etc.)
    metadata : Dict[str, str]
        Metadata fields to add
    """
    content = skill_path.read_text()
    frontmatter, body = parse_skill_frontmatter(skill_path)

    # Add agent-specific fields
    if "metadata" not in frontmatter:
        frontmatter["metadata"] = {}

    agent_key = f"{agent}_code" if agent == "claude" else agent
    if agent_key not in frontmatter["metadata"]:
        frontmatter["metadata"][agent_key] = {}

    frontmatter["metadata"][agent_key].update(metadata)

    # Reconstruct file
    new_content = "---\n"
    new_content += yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    new_content += "---\n\n"
    new_content += body

    skill_path.write_text(new_content)
