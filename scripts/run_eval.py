from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate retrieval metrics against acceptance thresholds")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds(path: Path) -> dict:
    thresholds: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", maxsplit=1)
        thresholds[key.strip()] = float(value.strip())
    return thresholds


def main() -> None:
    args = parse_args()
    metrics = load_json(args.predictions)
    thresholds = load_thresholds(args.thresholds)

    failed = []
    for metric_name, threshold in thresholds.items():
        value = float(metrics.get(metric_name, 0.0))
        ok = value >= threshold
        status = "PASS" if ok else "FAIL"
        print(f"{status} {metric_name}: value={value:.4f} threshold={threshold:.4f}")
        if not ok:
            failed.append(metric_name)

    if failed:
        raise SystemExit(f"Threshold check failed for: {', '.join(failed)}")

    print("All thresholds passed.")


if __name__ == "__main__":
    main()
