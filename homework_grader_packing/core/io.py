import json
from pathlib import Path
from typing import Dict, Any

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    path.write_bytes(data)

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
