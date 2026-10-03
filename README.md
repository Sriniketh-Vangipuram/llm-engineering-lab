
# LLM Engineering Lab

An engineering-first laboratory for building, measuring, and optimizing LLM systems.

The goal is to progress from using pretrained models to engineering reliable model,
inference, evaluation, and serving systems under real-world constraints.

## Engineering Objectives

- Tokenization and multilingual token efficiency
- Model architecture, memory, and inference fundamentals
- Training, continued pretraining, and supervised fine-tuning
- LoRA, QLoRA, quantization, and evaluation
- Prefill, decode, KV cache, and batching
- Inference APIs, model routing, and request management
- Observability, reliability, performance, and cost engineering

## Engineering Workflow

Each experiment should follow:

Problem -> Hypothesis -> Baseline -> Implementation -> Measurement
       -> Failure Analysis -> Optimization -> Conclusion

Results should be reproducible and should distinguish measured facts from
assumptions, estimates, and simulation results.

## Repository Structure

```text
llm-engineering-lab/
├── docs/                 # Architecture, decisions, reports, interview cases
├── experiments/          # Reproducible engineering experiments
├── src/
│   └── llm_engineering_lab/
│       ├── inference/    # Request, batch, scheduler, and engine simulation
│       └── tokenization/ # Tokenizer utilities and benchmarks
├── tests/                # Automated regression tests
├── pyproject.toml        # Package metadata and development tooling
└── README.md
```

The repository evolves incrementally. Components are promoted from experiments
into reusable source modules when their interfaces and behavior are understood.

## Requirements

- Python 3.11 or later
- Git
- Internet access when downloading models or tokenizers
- Optional GPU hardware for compatible model training and inference experiments

CPU-only development is supported for unit tests, configuration, API development,
and lightweight experiments.

## Installation

From the repository root:

```bat
python -m pip install -e ".[dev]"
```

This installs the project in editable mode with the development dependencies
defined in `pyproject.toml`.

## Validation

Run the automated test suite:

```bat
python -m pytest -v
```

Check Python code quality:

```bat
python -m ruff check src tests
```

Tests should be deterministic wherever possible and should not require a GPU or
external model downloads unless explicitly marked as integration tests.

## Experiments

Experiments are organized by topic and numbered as the project evolves.

Current areas include:

- Tokenization efficiency
- Token expansion and multilingual token distributions
- Token-aware request scheduling

Inspect each experiment's README or entry-point script for its exact execution
instructions, inputs, dependencies, and output artifacts.

## Current Implementation Boundaries

The inference scheduler currently models request admission, prompt prefill,
token-by-token decode, and request completion.

It is a simulation, not a production model-serving engine. Its estimated service
time is illustrative, and its output must not be interpreted as measured GPU
latency, throughput, or production SLO compliance.

Actual model inference, KV-cache measurements, serving-engine comparisons,
load testing, and production observability will be developed in later stages.

## Experiment Record

For meaningful experiments, record:

- Experiment identifier and question
- Hypothesis and baseline
- Model, tokenizer, dataset, and versions
- Hardware and software environment
- Configuration and random seed, when applicable
- Metrics, results, and unexpected behavior
- Failure analysis, trade-offs, and conclusions

## Engineering Principle

The objective is not to reproduce every component from scratch.

Use established libraries for mature infrastructure, build small controlled
experiments to understand their mechanisms, and implement custom components
when doing so answers a measurable engineering question.