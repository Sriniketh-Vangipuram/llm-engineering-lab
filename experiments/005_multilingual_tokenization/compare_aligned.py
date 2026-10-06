from __future__ import annotations

from corpus import LANGUAGE_SAMPLES

from llm_engineering_lab.tokenization.benchmark import load_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    tokenizer = load_tokenizer(MODEL_NAME, use_fast=True)

    languages = list(LANGUAGE_SAMPLES.keys())
    sample_count = len(LANGUAGE_SAMPLES[languages[0]])

    print("=" * 110)
    print("ALIGNED MULTILINGUAL TOKEN EFFICIENCY")
    print("=" * 110)
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Aligned sentences: {sample_count}")
    print()

    totals = {
        language: {
            "tokens": 0,
            "characters": 0,
            "utf8_bytes": 0,
            "words": 0,
        }
        for language in languages
    }

    for index in range(sample_count):
        print(f"Sentence {index + 1}")
        print("-" * 110)

        for language in languages:
            text = LANGUAGE_SAMPLES[language][index]

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )

            tokens = len(token_ids)
            characters = len(text)
            utf8_bytes = len(text.encode("utf-8"))
            words = len(text.split())

            totals[language]["tokens"] += tokens
            totals[language]["characters"] += characters
            totals[language]["utf8_bytes"] += utf8_bytes
            totals[language]["words"] += words

            print(
                f"{language:<12} "
                f"tokens={tokens:>4} "
                f"chars={characters:>4} "
                f"tok/char={tokens / characters:.4f} "
                f"tok/byte={tokens / utf8_bytes:.4f} "
                f"tok/word={tokens / words:.4f}"
            )

        print()

    print("=" * 110)
    print("AGGREGATED NORMALIZED EFFICIENCY")
    print("=" * 110)

    english_tokens_per_char = (
        totals["English"]["tokens"] / totals["English"]["characters"]
    )

    for language in languages:
        data = totals[language]

        tokens = data["tokens"]
        characters = data["characters"]
        utf8_bytes = data["utf8_bytes"]
        words = data["words"]

        tok_per_char = tokens / characters
        tok_per_byte = tokens / utf8_bytes
        tok_per_word = tokens / words

        print(
            f"{language:<12} "
            f"tokens={tokens:>5} "
            f"chars={characters:>5} "
            f"tok/char={tok_per_char:.4f} "
            f"tok/byte={tok_per_byte:.4f} "
            f"tok/word={tok_per_word:.4f} "
            f"chars/token={characters / tokens:.4f} "
            f"bytes/token={utf8_bytes / tokens:.4f} "
            f"vs-English={tok_per_char / english_tokens_per_char:.2f}x"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()