from pathlib import Path


def load_text(
    path: str | Path
) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()