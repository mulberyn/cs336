from pathlib import Path

def load_text(
    path: str | Path
) -> str:
    """读取 UTF‑8 文本文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()