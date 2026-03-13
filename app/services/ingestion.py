from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_records(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        out.append(
            {
                "doc_id": str(row.get("doc_id", f"doc-{i}")),
                "text": text,
                "modality": str(row.get("modality", "text")),
                "metadata": row.get("metadata", {}),
            }
        )
    return out
