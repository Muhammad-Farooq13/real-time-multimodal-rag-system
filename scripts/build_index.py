from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ingestion import load_jsonl, normalize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized JSON index for local retrieval")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL corpus")
    parser.add_argument("--output", type=Path, required=True, help="Path to output index JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    records = normalize_records(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
