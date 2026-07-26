from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN_FILENAMES = {
    "ultrasoundablations.py",
    "ultrasoundfinalmodels.py",
    "ultrasoundgeneralization.py",
}
ESSENTIAL_MARKDOWN = {
    "README.md",
    "MODEL_CARD.md",
}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "data",
    "dist",
    "htmlcov",
    "results",
    "venv",
}
TEXT_SUFFIXES = {
    ".cff",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PLACEHOLDER_PATTERNS = {
    "placeholder repository identifier": re.compile(
        r"<(?:owner|repository|repository-url)>", re.IGNORECASE
    ),
    "unfinished marker": re.compile(r"\b(?:TODO|FIXME|CHANGEME)\b"),
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _excluded(relative: Path) -> bool:
    return any(
        part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
        for part in relative.parts
    )


def _broken_local_links(path: Path, text: str) -> list[str]:
    broken = []
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group(1).strip().split("#", maxsplit=1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        linked = (path.parent / target).resolve()
        if not linked.exists():
            broken.append(target)
    return broken


def main() -> None:
    parser = argparse.ArgumentParser(description="Check public-release repository hygiene")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    problems = []

    if (root / "docs").exists():
        problems.append("Unexpected docs directory is present")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _excluded(relative):
            continue
        if path.name.lower() in FORBIDDEN_FILENAMES:
            problems.append(f"Original monolithic source is present: {relative}")
        if path.suffix.lower() == ".ipynb":
            problems.append(f"Notebook files are not part of the clean release: {relative}")
        if path.suffix.lower() == ".md" and relative.as_posix() not in ESSENTIAL_MARKDOWN:
            problems.append(f"Unexpected Markdown document: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            problems.append(f"Unexpected file larger than 10 MiB: {relative}")
        if path.suffix.lower() == ".py" and path.stat().st_size > 80_000:
            problems.append(f"Unexpectedly large Python file requires review: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "Dockerfile",
            "LICENSE",
            "Makefile",
        }:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if "\ufffd" in text or "\u00c2" in text or "\u00e2\u20ac" in text:
            problems.append(f"Encoding corruption detected in {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label} detected in {relative}")
        if relative.as_posix() != "scripts/check_release_hygiene.py":
            for label, pattern in PLACEHOLDER_PATTERNS.items():
                if pattern.search(text):
                    problems.append(f"{label} detected in {relative}")
        if path.suffix.lower() == ".md":
            for link in _broken_local_links(path, text):
                problems.append(f"Broken local Markdown link in {relative}: {link}")

    if problems:
        raise SystemExit("\n".join(problems))
    print(
        "Release hygiene passed: essential Markdown only; no notebooks, original "
        "scripts, placeholders, broken local links, large artifacts, mojibake, "
        "or detected secrets."
    )


if __name__ == "__main__":
    main()
