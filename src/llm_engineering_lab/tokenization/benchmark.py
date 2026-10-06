
"""Reusable, validated tokenizer benchmarking utilities."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
    use_fast: bool = True,
) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer with an explicit model revision option."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=use_fast,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer {model_name!r}. "
            "Check the model ID, network access, and Hugging Face cache."
        ) from exc

    return tokenizer


def _valid_context_limit(tokenizer: PreTrainedTokenizerBase) -> int | None:
    """Return a plausible tokenizer context limit, not an HF sentinel value."""
    limit = getattr(tokenizer, "model_max_length", None)

    if isinstance(limit, int) and 0 < limit < 10_000_000:
        return limit

    return None


def benchmark_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    language: str,
    *,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Measure tokenization, text size, and context consumption for one input."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")

    if not isinstance(add_special_tokens, bool):
        raise TypeError("add_special_tokens must be a boolean")

    start = perf_counter()
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    elapsed_seconds = perf_counter() - start

    # Count the same text with the tokenizer's normal special-token policy
    # to quantify the overhead introduced by special tokens.
    if add_special_tokens:
        token_ids_without_specials = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        special_token_count = len(token_ids) - len(token_ids_without_specials)
    else:
        token_ids_with_specials = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        special_token_count = len(token_ids_with_specials) - len(token_ids)

    characters = len(text)
    words = len(text.split())
    token_count = len(token_ids)
    utf8_bytes = len(text.encode("utf-8"))
    context_limit = _valid_context_limit(tokenizer)

    return {
        "language": language,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "words_whitespace_split": words,
        "tokens": token_count,
        "tokens_per_character": token_count / characters if characters else 0.0,
        "tokens_per_word": token_count / words if words else 0.0,
        "tokens_per_utf8_byte": token_count / utf8_bytes if utf8_bytes else 0.0,
        "special_tokens_included": add_special_tokens,
        "special_token_count": special_token_count,
        "tokenization_seconds": elapsed_seconds,
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_size": len(tokenizer),
        "context_limit": context_limit,
        "context_utilization": (
            token_count / context_limit
            if context_limit is not None
            else None
        ),
        "exceeds_context_limit": (
            token_count > context_limit
            if context_limit is not None
            else None
        ),
        "contains_replacement_character": "\ufffd" in text,
    }