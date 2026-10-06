from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from corpus import LANGUAGE_SAMPLES

from llm_engineering_lab.config import load_experiment_config
from llm_engineering_lab.experiments.runner import initialize_run
from llm_engineering_lab.tokenization.benchmark import (
    benchmark_text,
    load_tokenizer,
)

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate an interpolated percentile without external dependencies."""

    if not values:
        raise ValueError("Cannot calculate percentile of an empty list.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return ordered[lower]

    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize(values: list[float]) -> dict[str, float]:
    """Return descriptive statistics for a numeric sample."""

    if not values:
        raise ValueError("Cannot summarize an empty list.")

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def validate_corpus() -> None:
    """Validate the controlled multilingual benchmark corpus."""

    expected_languages = {
        "English",
        "Hindi",
        "Telugu",
        "Tamil",
        "Malayalam",
        "Odia",
    }

    actual_languages = set(LANGUAGE_SAMPLES)

    if actual_languages != expected_languages:
        raise ValueError(
            f"Unexpected languages. Expected {sorted(expected_languages)}, "
            f"got {sorted(actual_languages)}."
        )

    for language, samples in LANGUAGE_SAMPLES.items():
        if len(samples) != 20:
            raise ValueError(
                f"{language} must contain exactly 20 samples; "
                f"found {len(samples)}."
            )

        for index, text in enumerate(samples):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{language} sample {index} is empty or invalid."
                )

            if "\ufffd" in text:
                raise ValueError(
                    f"{language} sample {index} contains the Unicode "
                    "replacement character."
                )


def aggregate_language_results(
    language: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-sentence measurements for one language."""

    token_counts = [float(result["tokens"]) for result in results]
    token_per_character = [
        float(result["tokens_per_character"])
        for result in results
    ]
    token_per_word = [
        float(result["tokens_per_word"])
        for result in results
    ]
    token_per_byte = [
        float(result["tokens_per_utf8_byte"])
        for result in results
    ]

    total_characters = sum(result["characters"] for result in results)
    total_bytes = sum(result["utf8_bytes"] for result in results)
    total_words = sum(result["words_whitespace_split"] for result in results)
    total_tokens = sum(result["tokens"] for result in results)

    return {
        "language": language,
        "samples": len(results),
        "total_characters": total_characters,
        "total_utf8_bytes": total_bytes,
        "total_whitespace_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_character": summarize(token_per_character),
        "tokens_per_word": summarize(token_per_word),
        "tokens_per_utf8_byte": summarize(token_per_byte),
        "sequence_length_tokens": summarize(token_counts),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    *,
    config: Any,
    tokenizer_metadata: dict[str, Any],
    language_summary: list[dict[str, Any]],
    english_token_count: int,
) -> None:
    """Write a human-readable benchmark report."""

    lines = [
        "# Multilingual Tokenization Benchmark",
        "",
        f"**Experiment:** `{config.experiment_id}`",
        "",
        f"**Tokenizer:** `{tokenizer_metadata['model_name']}`",
        "",
        "## Tokenization summary",
        "",
       (
            "| Language | Samples | Tokens | Tok/Char Mean | Tok/Char P95 | "
            "Tok/Word Mean | Tok/Byte Mean | English-relative tokens |"
       ),
    ]

    for summary in language_summary:
        english_relative = summary["total_tokens"] / english_token_count

        lines.append(
            f"| {summary['language']} "
            f"| {summary['samples']} "
            f"| {summary['total_tokens']} "
            f"| {summary['tokens_per_character']['mean']:.4f} "
            f"| {summary['tokens_per_character']['p95']:.4f} "
            f"| {summary['tokens_per_word']['mean']:.4f} "
            f"| {summary['tokens_per_utf8_byte']['mean']:.4f} "
            f"| {english_relative:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Tokenizer metadata",
            "",
            f"- Tokenizer class: `{tokenizer_metadata['tokenizer_class']}`",
            f"- Fast tokenizer: `{tokenizer_metadata['is_fast']}`",
            f"- Vocabulary size: `{tokenizer_metadata['vocabulary_size']}`",
            f"- Model max length metadata: `{tokenizer_metadata['model_max_length']}`",
            f"- Special tokens map: `{tokenizer_metadata['special_tokens_map']}`",
            "",
            "## Interpretation",
            "",
            "The English-relative value compares each language's total token "
            "count against the English corpus total.",
            "",
            "These measurements describe this controlled corpus. They should "
            "not be interpreted as universal language-level tokenization ratios.",
            "",
            "Tokenization latency is recorded at sample level, but these short "
            "sample timings are not treated as a production latency benchmark.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    validate_corpus()

    config = load_experiment_config(CONFIG_PATH)
    run_dir = initialize_run(config)

    tokenizer_name = str(
        config.parameters.get(
            "tokenizer_name",
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
    )

    revision = config.parameters.get("revision")

    tokenizer = load_tokenizer(
        tokenizer_name,
        revision=revision,
        use_fast=True,
    )

    tokenizer_metadata = {
        "model_name": tokenizer_name,
        "revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocabulary_size": len(tokenizer),
        "model_max_length": getattr(
            tokenizer,
            "model_max_length",
            None,
        ),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }

    sample_results: list[dict[str, Any]] = []
    language_summary: list[dict[str, Any]] = []

    for language, samples in LANGUAGE_SAMPLES.items():
        language_results = []

        for index, text in enumerate(samples):
            result = benchmark_text(
                tokenizer,
                text,
                language,
                add_special_tokens=False,
            )

            result["sample_index"] = index
            result["text"] = text

            language_results.append(result)
            sample_results.append(result)

        language_summary.append(
            aggregate_language_results(
                language,
                language_results,
            )
        )

    english_summary = next(
        summary
        for summary in language_summary
        if summary["language"] == "English"
    )

    metrics = {
        "experiment": {
            "id": config.experiment_id,
            "description": config.description,
        },
        "tokenizer": tokenizer_metadata,
        "corpus": {
            "languages": list(LANGUAGE_SAMPLES.keys()),
            "samples_per_language": 20,
            "total_samples": len(sample_results),
        },
        "language_summary": language_summary,
        "interpretation": {
            "english_reference_tokens": english_summary["total_tokens"],
            "english_relative_ratio_definition": (
                "language_total_tokens / english_total_tokens"
            ),
        },
    }

    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "sample_results.json",
        {"results": sample_results},
    )

    write_report(
        run_dir / "report.md",
        config=config,
        tokenizer_metadata=tokenizer_metadata,
        language_summary=language_summary,
        english_token_count=english_summary["total_tokens"],
    )

    run_metadata_path = run_dir / "run.json"
    run_metadata = json.loads(
        run_metadata_path.read_text(encoding="utf-8")
    )

    run_metadata["status"] = "completed"
    run_metadata["started_at"] = run_metadata["created_at"]
    run_metadata["completed_at"] = run_metadata["created_at"]

    write_json(run_metadata_path, run_metadata)

    print("Benchmark completed.")
    print(f"Run directory: {run_dir}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Samples: {len(sample_results)}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()