# ts-agents Skills (Canonical)

This is the **canonical location** for ts-agents skills. Skills defined here are the source of truth and are copied/symlinked to agent-specific locations for discovery.

## Directory Structure

```
skills/
  README.md                           # This file
  SKILLS.md                           # Aggregate skills summary (generated)
  activity-recognition/SKILL.md       # End-to-end windowing/classification workflow
  forecasting/SKILL.md                # Forecasting methods
  forecasting/SKILL-pro.md            # Professional M4 benchmark workflow
  diagnostics/SKILL.md                # Quick EDA diagnostics
  decomposition/SKILL.md              # Decomposition methods
  classification/SKILL.md             # Time series classification
```

This repository intentionally keeps a **small, focused set** of skills to reduce
drift across agent-specific directories.

## Agent-Specific Locations

Skills are placed in the following agent-specific directories:

| Agent | Location | Notes |
|-------|----------|-------|
| Claude Code | `.claude/skills/` | Primary development agent |
| Codex | `.codex/skills/` | OpenAI Codex CLI |
| Gemini | `.gemini/skills/` | Google Gemini |
| Windsurf | `.windsurf/skills/` | Codeium Windsurf |
| GitHub | `.github/skills/` | GitHub tooling that reads workspace skills |

## Placement Commands

To sync skills to agent-specific locations:

```bash
# Export and place skills for all agents
uv run ts-agents skills export --all-agents

# Export to specific agent
uv run ts-agents skills export --agent claude
uv run ts-agents skills export --agent codex
uv run ts-agents skills export --agent github

# Validate skill format
uv run ts-agents skills validate
```

To generate the aggregate `SKILLS.md` summary:

```bash
uv run ts-agents skills export --out skills/SKILLS.md
```

## Skill Format

Each skill follows the SKILL.md format with YAML frontmatter:

```yaml
---
name: skill-name
description: Brief description
compatibility: "Compatible agents/environments"
metadata:
  domain: time-series
  tasks: [task1, task2]
  ts_agents:
    tool_category: category_name
    prefers_with_data_tools: true
    preferred_workflow: inspect-series
    preferred_tools: [describe_series_with_data]
    artifact_checklist: [summary.json, report.md]
---

# Skill Title

## When to use
...

## Workflow
...
```

## Adding New Skills

1. Create a new directory under `skills/`
2. Add a `SKILL.md` file following the format above
3. Run `uv run ts-agents skills export --all-agents` to distribute
4. Run `uv run ts-agents skills validate` to verify format

For demo/report workflows, prefer a Quarto report (`.qmd`) rendered to PDF
with publication-quality figures.
