from __future__ import annotations

from pathlib import Path

from corpus import LANGUAGE_SAMPLES

from llm_engineering_lab.tokenization.benchmark import load_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR = Path("artifacts") / "tokenization-segmentation"


def main() -> None:
    tokenizer = load_tokenizer(
        MODEL_NAME,
        use_fast=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("TOKEN-LEVEL SEGMENTATION INSPECTION")
    print("=" * 100)
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Class: {type(tokenizer).__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print()

    for language, samples in LANGUAGE_SAMPLES.items():
        text = samples[0]

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        decoded_pieces = [
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ]

        print("-" * 100)
        print(f"LANGUAGE: {language}")
        print(f"TEXT: {text}")
        print(f"CHARACTERS: {len(text)}")
        print(f"TOKENS: {len(token_ids)}")
        print()

        print("TOKEN TABLE")
        print(f"{'INDEX':>5}  {'TOKEN_ID':>8}  {'TOKEN_STRING':<30}  DECODED")
        print("-" * 100)

        for index, (token_id, token_string, decoded) in enumerate(
            zip(token_ids, token_strings, decoded_pieces)
        ):
            print(
                f"{index:>5}  "
                f"{token_id:>8}  "
                f"{token_string!r:<30}  "
                f"{decoded!r}"
            )

        print()

    print("=" * 100)
    print("Inspection complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()