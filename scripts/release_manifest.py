from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

INCLUDED_SUFFIXES = {
    ".cff",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
INCLUDED_NAMES = {
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "Makefile",
}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "data",
    "dist",
    "htmlcov",
    "results",
    "venv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return path.name in INCLUDED_NAMES or path.suffix.lower() in INCLUDED_SUFFIXES


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-checkout"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a release reproducibility manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="release_manifest.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and _included(path, root)
    }
    payload = {
        "git_commit": _git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "files": files,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
