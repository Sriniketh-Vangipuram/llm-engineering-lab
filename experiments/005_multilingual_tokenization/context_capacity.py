from __future__ import annotations

from corpus import LANGUAGE_SAMPLES

from llm_engineering_lab.tokenization.benchmark import load_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_BUDGETS = [512, 1024, 2048, 4096]


def count_tokens(tokenizer, text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def tokens_to_fill_budget(tokenizer, samples: list[str], budget: int) -> dict:
    repeated_text = ""
    repetitions = 0

    while True:
        candidate = repeated_text + (" " if repeated_text else "") + samples[
            repetitions % len(samples)
        ]

        token_count = count_tokens(tokenizer, candidate)

        if token_count > budget:
            break

        repeated_text = candidate
        repetitions += 1

    final_tokens = count_tokens(tokenizer, repeated_text)

    return {
        "tokens": final_tokens,
        "characters": len(repeated_text),
        "words": len(repeated_text.split()),
        "repetitions": repetitions,
        "utilization": final_tokens / budget if budget else 0.0,
    }


def main() -> None:
    tokenizer = load_tokenizer(MODEL_NAME, use_fast=True)

    print("=" * 110)
    print("MULTILINGUAL CONTEXT CAPACITY EXPERIMENT")
    print("=" * 110)
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Context budgets: {CONTEXT_BUDGETS}")
    print()

    for budget in CONTEXT_BUDGETS:
        print(f"CONTEXT BUDGET: {budget} TOKENS")
        print("-" * 110)

        for language, samples in LANGUAGE_SAMPLES.items():
            result = tokens_to_fill_budget(
                tokenizer,
                samples,
                budget,
            )

            print(
                f"{language:<12} "
                f"tokens={result['tokens']:>5} "
                f"chars={result['characters']:>6} "
                f"words={result['words']:>5} "
                f"repetitions={result['repetitions']:>4} "
                f"utilization={result['utilization']:.2%}"
            )

        print()

    print("=" * 110)
    print("INTERPRETATION")
    print("=" * 110)
    print(
        "Higher character/word counts at the same token budget indicate "
        "greater effective text capacity for that language."
    )
    print("=" * 110)


if __name__ == "__main__":
    main()