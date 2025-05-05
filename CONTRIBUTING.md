# Contributing to llm-eval-harness

Thanks for your interest in contributing. This document covers the basics — issue triage, pull requests, and the local dev loop.

## Code of Conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the maintainers.

## Reporting issues

- Search existing issues before opening a new one.
- Use the provided templates. Include: `llm-eval --version`, Python version, OS, a minimal reproducer, and the actual vs expected output.

## Pull requests

1. Open an issue first for non-trivial changes.
2. Fork the repo and create a branch from `main`.
3. Install dev extras: `make install`.
4. Add or update tests under `tests/`. Coverage must stay ≥ 90%.
5. Run `make format lint test` locally.
6. Keep PRs focused — one logical change per PR.
7. Update `CHANGELOG.md` under "Unreleased".
8. Sign off your commits (`git commit -s`).

## Style

- Formatter: `black` with `--line-length=100`.
- Imports: `isort --profile=black`.
- Linter: `ruff` (E, F, W, I, B, UP, SIM, RUF).
- Type checker: `mypy --strict` on `src/`.

## Project layout

```
src/llm_eval_harness/   # package
tests/                  # mirrors the package layout
config/                 # example YAML profiles
examples/               # runnable end-to-end scripts
docs/                   # design docs
```

## Adding a new scorer

1. Implement the callable in `src/llm_eval_harness/scorers/`.
2. Register it via `@registry.scorer("name")`.
3. Add tests under `tests/scorers/`.
4. Document it in `README.md` (Features) and `docs/ARCHITECTURE.md`.

## Adding a new judge adapter

1. Subclass `llm_eval_harness.scorers.judge.Judge` in `src/llm_eval_harness/scorers/judge/adapters/`.
2. Implement `async judge(prompt: str, candidate: str, reference: str | None) -> Verdict`.
3. Add tests under `tests/scorers/judge/`.

## Releasing

1. Bump version in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Tag: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
4. Push: `git push origin vX.Y.Z`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.