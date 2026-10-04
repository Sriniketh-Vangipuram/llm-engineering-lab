
import statistics
import time

import torch

from llm_engineering_lab.inference.pretrained import PretrainedGenerator

MODEL_ID = "gpt2"
PROMPT = (
    "When designing a production language model inference system, "
    "the most important performance considerations are"
)
MAX_NEW_TOKENS = 40


def synchronize_if_needed(device: torch.device) -> None:
    """Ensure GPU operations finish before recording elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    generator = PretrainedGenerator(model_id=MODEL_ID, device="cpu")
    tokenizer = generator.tokenizer
    model = generator.model
    device = generator.device

    # Measure tokenization separately from model execution.
    tokenization_start = time.perf_counter()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokenization_seconds = time.perf_counter() - tokenization_start

    prompt_tokens = input_ids.shape[1]

    # Prefill: process the entire prompt and prepare the KV cache.
    synchronize_if_needed(device)
    prefill_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)

        # Greedy selection of the first output token.
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        past_key_values = outputs.past_key_values

    synchronize_if_needed(device)
    first_token_available = time.perf_counter()

    model_ttft_seconds = first_token_available - prefill_start
    generated_ids = [next_token]
    decode_step_seconds = []

    # Decode: process one new token at a time, reusing the KV cache.
    with torch.inference_mode():
        for _ in range(1, MAX_NEW_TOKENS):
            if (
                tokenizer.eos_token_id is not None
                and next_token.item() == tokenizer.eos_token_id
            ):
                break

            synchronize_if_needed(device)
            step_start = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            past_key_values = outputs.past_key_values

            synchronize_if_needed(device)
            step_end = time.perf_counter()

            decode_step_seconds.append(step_end - step_start)
            generated_ids.append(next_token)

    generated_tokens = len(generated_ids)
    decode_seconds = sum(decode_step_seconds)
    total_model_seconds = model_ttft_seconds + decode_seconds

    generated_tensor = torch.cat(generated_ids, dim=1)
    full_ids = torch.cat([input_ids, generated_tensor], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    print("\n=== Token-Level Inference Report ===")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokenization time: {tokenization_seconds:.4f} s")
    print(f"Prefill + first-token selection: {model_ttft_seconds:.4f} s")
    print(f"Decode time after first token: {decode_seconds:.4f} s")
    print(f"Total model time: {total_model_seconds:.4f} s")

    if decode_step_seconds:
        print(
            "Mean decode-step latency: "
            f"{statistics.mean(decode_step_seconds) * 1000:.2f} ms"
        )
        print(
            "Median decode-step latency: "
            f"{statistics.median(decode_step_seconds) * 1000:.2f} ms"
        )
        print(
            "Mean decode throughput: "
            f"{1 / statistics.mean(decode_step_seconds):.2f} tokens/s"
        )

    print("\n=== Generated Text ===")
    print(text)


if __name__ == "__main__":
    main()