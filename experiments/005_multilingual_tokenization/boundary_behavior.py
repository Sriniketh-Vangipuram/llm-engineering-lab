from __future__ import annotations

from llm_engineering_lab.tokenization.benchmark import load_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BUDGETS = [128, 512, 1024]


def count_tokens(tokenizer, text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def main() -> None:
    tokenizer = load_tokenizer(MODEL_NAME, use_fast=True)

    base = (
        "Large language models process text as sequences of tokens. "
        "Tokenization determines how much context a model can consume, "
        "how much memory inference requires, and how much computation "
        "is performed during prefill."
    )

    print("=" * 100)
    print("TOKENIZER CONTEXT-BOUNDARY BEHAVIOR")
    print("=" * 100)
    print(f"Tokenizer: {MODEL_NAME}")
    print()

    for budget in BUDGETS:
        text = ""
        while count_tokens(tokenizer, text) < budget:
            candidate = text + (" " if text else "") + base
            if count_tokens(tokenizer, candidate) > budget:
                break
            text = candidate

        below_tokens = count_tokens(tokenizer, text)

        one_more = text + " " + base
        above_tokens = count_tokens(tokenizer, one_more)

        print(f"Budget: {budget}")
        print(f"  Below boundary : {below_tokens} tokens")
        print(f"  Above boundary : {above_tokens} tokens")
        print(f"  Overflow       : {above_tokens > budget}")

        truncated = tokenizer(
            one_more,
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
        )

        truncated_tokens = len(truncated["input_ids"])

        print(f"  Truncated      : {truncated_tokens} tokens")
        print(f"  Exact budget   : {truncated_tokens == budget}")
        print()


if __name__ == "__main__":
    main()