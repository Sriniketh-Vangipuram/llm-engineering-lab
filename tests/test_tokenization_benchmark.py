
import pytest

from llm_engineering_lab.tokenization.benchmark import benchmark_text


class FakeTokenizer:
    """Deterministic tokenizer for unit tests; no model download required."""

    model_max_length = 8

    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert truncation is False
        tokens = text.split() if text else []
        if add_special_tokens:
            return [101, *range(len(tokens)), 102]
        return list(range(len(tokens)))


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def test_basic_tokenization_metrics(tokenizer):
    result = benchmark_text(tokenizer, "hello world", "English")

    assert result["tokens"] == 2
    assert result["characters"] == 11
    assert result["words_whitespace_split"] == 2
    assert result["tokens_per_word"] == 1.0
    assert result["tokens_per_character"] == pytest.approx(2 / 11)
    assert result["vocabulary_size"] == 100
    assert result["tokenizer_class"] == "FakeTokenizer"


def test_special_token_overhead(tokenizer):
    result = benchmark_text(
        tokenizer,
        "hello world",
        "English",
        add_special_tokens=True,
    )

    assert result["tokens"] == 4
    assert result["special_tokens_included"] is True
    assert result["special_token_count"] == 2


def test_special_token_overhead_when_not_included(tokenizer):
    result = benchmark_text(tokenizer, "hello world", "English")

    assert result["tokens"] == 2
    assert result["special_tokens_included"] is False
    assert result["special_token_count"] == 2


def test_empty_text(tokenizer):
    result = benchmark_text(tokenizer, "", "English")

    assert result["characters"] == 0
    assert result["utf8_bytes"] == 0
    assert result["words_whitespace_split"] == 0
    assert result["tokens"] == 0
    assert result["tokens_per_character"] == 0.0
    assert result["tokens_per_word"] == 0.0
    assert result["tokens_per_utf8_byte"] == 0.0


def test_context_utilization_and_overflow(tokenizer):
    within_limit = benchmark_text(tokenizer, "one two three", "English")
    over_limit = benchmark_text(
        tokenizer,
        "one two three four five six seven eight nine",
        "English",
    )

    assert within_limit["context_limit"] == 8
    assert within_limit["context_utilization"] == pytest.approx(3 / 8)
    assert within_limit["exceeds_context_limit"] is False

    assert over_limit["tokens"] == 9
    assert over_limit["context_utilization"] == pytest.approx(9 / 8)
    assert over_limit["exceeds_context_limit"] is True


def test_utf8_byte_count_and_replacement_detection(tokenizer):
    result = benchmark_text(tokenizer, "नमस्ते �", "Hindi")

    assert result["utf8_bytes"] == len("नमस्ते �".encode("utf-8"))
    assert result["contains_replacement_character"] is True


def test_invalid_text_type(tokenizer):
    with pytest.raises(TypeError, match="text must be a string"):
        benchmark_text(tokenizer, 123, "English")


@pytest.mark.parametrize("language", ["", "   ", None])
def test_invalid_language(tokenizer, language):
    with pytest.raises(ValueError, match="language must be a non-empty string"):
        benchmark_text(tokenizer, "hello", language)


def test_invalid_special_token_option(tokenizer):
    with pytest.raises(TypeError, match="add_special_tokens must be a boolean"):
        benchmark_text(tokenizer, "hello", "English", add_special_tokens=1)