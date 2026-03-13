# Changelog

## v0.1.2 - 2026-03-13

- added an MIT license for safe public GitHub publication
- updated GitHub repository metadata and live repository description
- aligned repository ownership metadata with the authenticated GitHub account
- fixed benchmark parsing for the current k6 JSON summary schema
- generated and prepared a committed local benchmark evidence run

## v0.1.0 - 2026-03-13

- inventoried the repository and validated the existing API and benchmark test suite
- split dependencies into runtime and CI requirements for safer cloud installs
- upgraded GitHub Actions to Python 3.11 and 3.12 with Ruff, coverage XML, Codecov, and modern Docker actions
- added Streamlit Cloud runtime files: `.python-version`, `runtime.txt`, `packages.txt`, and `.streamlit/config.toml`
- created `train_demo.py` with deterministic synthetic data generation, multi-model training, and persisted bundle output
- added self-healing bundle loading so missing or corrupt model artifacts are rebuilt automatically
- created `streamlit_app.py` with five tabs: Overview, Model Results, Analytics, Pipeline/API, and Predict
- added live prediction with probability distribution, confidence, risk band, and input summary
- added tests for benchmark parsing, metrics, and demo bundle rebuild behavior
- refreshed README and repository metadata for GitHub publication readiness
