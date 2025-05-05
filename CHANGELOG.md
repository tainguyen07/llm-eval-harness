# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pairwise comparison with Bradley–Terry scoring and win-rate confidence intervals.
- Webhook notifier for regression gates (Slack-compatible JSON payload).
- `gates` CLI subcommand with YAML-driven thresholds.

### Changed
- Async runner defaults to `concurrency=8` (was 4).

### Fixed
- HTML report embeds JSON correctly when run ID contains spaces.

## [0.4.2] — 2025-07-14

### Added
- `llm_eval_harness.reporting.markdown` module with summary writer.
- `scipy` dependency for statistical tests in pairwise comparison.

### Fixed
- Retry decorator no longer swallows `KeyboardInterrupt`.
- CSV report writer escapes double quotes correctly.

## [0.4.1] — 2025-05-29

### Fixed
- Prompt template fallback raises `PromptNotFoundError` with full lookup chain.

## [0.4.0] — 2025-04-02

### Added
- Prompt registry with content-addressed versions and rollback.
- JSON-schema scorer with `jsonschema` validation.
- Pytest plugin (`llm_eval_harness.integrations.pytest_plugin`).

### Changed
- `run()` now returns a typed `RunReport` instead of a dict.

### Removed
- Legacy `RunnerV1` class (deprecated in 0.3).

## [0.3.0] — 2025-02-10

### Added
- Async runner with bounded concurrency (`asyncio.Semaphore`).
- LLM-as-judge scorer with structured verdicts.

## [0.2.0] — 2024-12-05

### Added
- Deterministic scorers: exact match, contains, regex, numeric tolerance, Levenshtein, token F1.

## [0.1.0] — 2024-10-18

### Added
- Initial release: dataset loader, runner, JSONL run store, HTML report.