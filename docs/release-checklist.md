# Release Checklist

Use this checklist before tagging a public milestone or sharing the repository in job applications.

## Quality

- Local index builds successfully
- Evaluation threshold check passes
- Test suite passes
- Docker image builds successfully
- README reflects the current feature set

## Performance

- Recent benchmark report exists under a representative configuration
- p95 and p99 latency are documented honestly
- Cache hit ratio is reported if cache is part of the performance claim
- Any throughput claim is backed by saved artifacts

## Documentation

- Architecture document is current
- System card is current
- Portfolio summary reflects the current system scope
- Setup steps are accurate on a clean environment

## Repository Hygiene

- Secrets are not committed
- `.env` is excluded
- Local-only artifacts are ignored or intentionally curated
- Pull request template and issue templates are present

## Public Presentation

- Pinned benchmark report or screenshot is ready if you are using this for applications
- Resume bullets align with actual implemented features
- GitHub Actions workflow status is green
- Demo flow is rehearsed and reproducible
