# Contributing Guide

## Purpose

This repository is structured as a portfolio-quality applied AI systems project. Contributions should preserve reproducibility, observability, and clear system behavior.

## Development Workflow

1. Create a feature branch from `main`.
2. Keep changes focused on one concern.
3. Run local validation before opening a pull request.
4. Include benchmark or evaluation evidence when changing retrieval, caching, or latency-sensitive logic.

## Local Validation

Run the following before submitting changes:

```powershell
python scripts/build_index.py --input data/raw/sample_docs.jsonl --output data/processed/index.json
python scripts/run_eval.py --predictions eval/sample_predictions.json --thresholds eval/thresholds.yaml
python -m pytest -q
```

If your change affects serving or performance behavior, also run:

```powershell
python scripts/run_benchmark.py --base-url http://localhost:8000 --rate 100 --duration 1m
```

## Code Expectations

- Prefer minimal, focused changes.
- Preserve API behavior unless the change explicitly updates the contract.
- Add tests for new behavior when practical.
- Maintain safe fallback behavior for optional infrastructure and external services.
- Keep documentation aligned with implementation.

## Pull Request Expectations

Each pull request should explain:

- what changed
- why it changed
- how it was validated
- what tradeoffs or risks remain

For performance-sensitive changes, include at least one of the following:

- benchmark report
- metrics snapshot
- trace evidence
- evaluation threshold output

## Areas Especially Worth Contributing To

- multimodal online inference paths for audio and vision
- tenant-aware retrieval and metadata filtering
- RAGAS-based answer quality evaluation
- more realistic corpora and retrieval benchmarks
- deployment hardening and security controls
