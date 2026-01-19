---
name: list_skills
description: 登録されているスキル一覧とその解説を表示する
---

# List Skills

This skill lists all available agent skills in the project.

## Instructions

1.  **Scan Directory**:
    *   List directories in `.agent/skills/`.

2.  **Extract Info**:
    *   For each directory, look for `SKILL.md`.
    *   Read the YAML frontmatter (lines between `---`) to extract `name` and `description`.

3.  **Display**:
    *   Output a formatted list or table of skills.
    *   Example format:
        *   `skill_name`: Description string
