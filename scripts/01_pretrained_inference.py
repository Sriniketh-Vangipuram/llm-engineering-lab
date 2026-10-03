
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "gpt2"
PROMPT = "A production language model"
MAX_NEW_TOKENS = 40


def main() -> None:
    print(f"Loading tokenizer: {MODEL_ID}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    load_seconds = time.perf_counter() - load_start
    device = next(model.parameters()).device

    inputs = tokenizer(PROMPT, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].shape[1]

    print(f"\nPrompt: {PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nGenerating...")

    generation_start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = time.perf_counter() - generation_start

    generated_tokens = output.shape[1] - prompt_tokens
    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    print("\n--- Results ---")
    print(f"Generated text: {generated_text}")
    print(f"Model load time: {load_seconds:.2f} seconds")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Generated tokens: {generated_tokens}")
    print(
        "Generation throughput: "
        f"{generated_tokens / generation_seconds:.2f} tokens/sec"
        if generation_seconds > 0
        else "Generation throughput: N/A"
    )


if __name__ == "__main__":
    main()