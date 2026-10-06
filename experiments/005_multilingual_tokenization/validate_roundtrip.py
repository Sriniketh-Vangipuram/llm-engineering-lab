from __future__ import annotations

from corpus import LANGUAGE_SAMPLES

from llm_engineering_lab.tokenization.benchmark import load_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    tokenizer = load_tokenizer(
        MODEL_NAME,
        use_fast=True,
    )

    print("=" * 100)
    print("TOKENIZER ROUND-TRIP VALIDATION")
    print("=" * 100)
    print(f"Tokenizer: {MODEL_NAME}")
    print()

    failures = 0

    for language, samples in LANGUAGE_SAMPLES.items():
        language_failures = 0

        for index, text in enumerate(samples):
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )

            decoded = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            matches = decoded == text

            if not matches:
                language_failures += 1
                failures += 1

                print("-" * 100)
                print(f"FAILURE: {language} sample {index}")
                print(f"Original : {text!r}")
                print(f"Decoded  : {decoded!r}")

        print(
            f"{language:<12} "
            f"samples={len(samples):>3} "
            f"failures={language_failures:>3}"
        )

    print()
    print("=" * 100)

    if failures == 0:
        print("RESULT: PASS")
        print("All samples survived encode → decode round-trip exactly.")
    else:
        print(f"RESULT: FAIL ({failures} samples)")
        raise SystemExit(1)

    print("=" * 100)


if __name__ == "__main__":
    main()