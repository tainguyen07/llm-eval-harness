# Architecture

`llm-eval-harness` is structured as a strict pipeline: dataset → runner → scorers → results → reporters → gates. Every stage is a typed object that can be inspected, serialized, or swapped.

## Module layout

```
src/llm_eval_harness/
├── cli.py               # Click CLI: run, diff, report, register, list, gates
├── config.py            # Pydantic configs: EvalConfig, RunSpec, JudgeConfig, RetryConfig
├── registry.py          # In-process registry for prompts/scorers/judges/datasets
│
├── core/                # Domain types and orchestration
│   ├── errors.py
│   ├── models.py        # Example, ExampleResult, RunReport, Verdict
│   └── run.py           # run() facade wiring everything together
│
├── datasets/            # Loaders and preprocessing
│   ├── loaders.py       # JSONL, CSV, dicts, Hugging Face
│   └── preprocess.py    # lowercase, normalize, truncate
│
├── prompts/             # Versioned prompt registry
│   └── registry.py      # Prompt, PromptRegistry, render, diff, rollback
│
├── runners/             # Concurrency primitives
│   ├── async_runner.py  # asyncio.Semaphore + bounded retry
│   ├── thread_runner.py # ThreadPoolExecutor for sync predictors
│   └── retry.py         # RetryPolicy with exponential backoff + jitter
│
├── scorers/             # Scoring primitives
│   ├── deterministic.py # exact_match, contains, regex, fuzzy, levenshtein, token_f1, numeric, json_schema
│   ├── judge.py         # Judge ABC + StubJudge, OpenAIJudge, AnthropicJudge
│   ├── pairwise.py      # Head-to-head with Wilson CI + Bradley–Terry
│   └── apply.py         # apply_scorers() across results
│
├── evaluation/          # CI primitives
│   ├── diff.py          # RegressionDiff between two runs
│   └── gates.py         # Evaluate min_*/max_* thresholds
│
├── reporting/           # Output writers
│   ├── html.py          # Jinja2 HTML report
│   ├── markdown.py      # Markdown summary
│   └── serialize.py     # JSON, CSV, JUnit
│
├── integrations/        # Outbound
│   ├── notifier.py      # WebhookNotifier + Slack formatter
│   └── pytest_plugin.py # pytest --llm-eval-config fixture
│
└── utils/               # Logging, hashing, stats, timing, tokens
```

## Data flow

1. `load_dataset(path)` parses JSONL/CSV/dicts into `Example` rows.
2. `render_template(prompt, inputs)` substitutes variables (Python `Template` or Jinja2).
3. The runner (`AsyncRunner` or `ThreadRunner`) calls `predict(example)` for each example, bounded by `concurrency` and retried per `RetryPolicy`.
4. Each result is wrapped in `ExampleResult(prediction, latency_ms, cost_usd, error)`.
5. `apply_scorers()` runs each registered scorer and attaches `ScorerOutput` entries.
6. The collected `RunReport` is serialized via `write_artifacts()` (HTML, Markdown, CSV, JSON, JUnit).
7. `diff(base, head)` and `evaluate_gates(report, gates)` enforce CI rules.

## Design choices

### Typed everywhere
Pydantic models for every boundary object make round-trips through JSON safe and IDE-friendly. `RunReport` can be loaded back from disk and re-diffed against a newer run.

### Async-first, with a sync escape hatch
`AsyncRunner` is the default. `ThreadRunner` is provided for synchronous predictors (legacy SDKs, locally-hosted models in thread-unsafe bindings).

### Registry over globals
Scorers, judges, and prompts are registered through decorators. Users can plug in their own scorers in two lines: implement, then `@registry.scorer("name")`.

### Reports are self-contained
HTML reports embed the JSON payload inline, so they round-trip without external assets.

### Gates are simple
No DSL — gates are a `dict[str, float]`. The keys describe the threshold (`min_exact_match`, `max_p99_ms`). Adding a new metric is one branch in `evaluate_gates()`.

## Trade-offs

- **No distributed runner.** The single-process runner is sufficient up to a few thousand examples per minute on commodity hardware. For larger jobs, integrate with Ray or Dask at the runner layer — the `Runner.stream()` async-iterator interface is the seam.
- **Heuristic tokenizer.** `count_tokens()` uses a 4-chars/token heuristic by default. Provide a tiktoken encoder for exact billing when cost matters.
- **Sync judges return only structured JSON.** Prompts that require tool use or chain-of-thought are out of scope — judges should be cheap and deterministic.

## Future work

- Distributed runner (Ray / Dask)
- Native Hugging Face `datasets` integration with streaming
- Web UI for browsing runs
- Calibration suite for LLM judges