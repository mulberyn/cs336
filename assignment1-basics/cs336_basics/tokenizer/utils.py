import mmap
from pathlib import Path

def load_text(
    path: str | Path
) -> str:
    path = Path(path)
    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode('utf-8')