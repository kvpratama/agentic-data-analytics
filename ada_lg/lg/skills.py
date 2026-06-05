"""Progressive-disclosure skills loader."""

from __future__ import annotations

from dataclasses import dataclass

from deepagents.backends.protocol import BackendProtocol
from langchain.tools import BaseTool, tool


@dataclass(frozen=True)
class _Skill:
    """Parsed SKILL.md content."""

    name: str
    description: str
    path: str
    body: str


def _parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Parse frontmatter and body from a SKILL.md file.

    Args:
        text: Raw markdown file content.

    Returns:
        A frontmatter dict and the markdown body after frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
    if end is None:
        return {}, text
    frontmatter: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip("\"'")
    body = "\n".join(lines[end + 1 :])
    return frontmatter, body


def _content_to_text(content: object) -> str:
    """Normalize backend file content to text."""
    if isinstance(content, list):
        return "\n".join(str(line) for line in content)
    return str(content)


def build_skills(backend: BackendProtocol, *, skill_dir: str) -> tuple[str, BaseTool]:
    """Scan a skill directory and return a prompt index plus read_skill tool.

    Args:
        backend: BackendProtocol with sync read/glob support.
        skill_dir: Absolute virtual directory to scan.

    Returns:
        Prompt index and a LangChain read_skill tool.
    """
    glob_result = backend.glob("**/SKILL.md", skill_dir)
    skills: dict[str, _Skill] = {}
    for match in glob_result.matches or []:
        match_path = match.get("path") or match.get("name")
        if not isinstance(match_path, str):
            continue
        path = (
            match_path if match_path.startswith("/") else skill_dir.rstrip("/") + "/" + match_path
        )
        read = backend.read(path)
        if read.error or read.file_data is None:
            continue
        frontmatter, body = _parse_skill_md(_content_to_text(read.file_data["content"]))
        name = frontmatter.get("name") or match_path
        description = frontmatter.get("description") or ""
        skills[name] = _Skill(name=name, description=description, path=path, body=body)

    index = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())
    if not index:
        index = "(no skills available)"

    @tool
    async def read_skill(name: str) -> str:
        """Return the full SKILL.md body for a named skill."""
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(skills) or "(none)"
            return f"Error: unknown skill {name!r}. Available: {available}"
        return skill.body

    return index, read_skill
