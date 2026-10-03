
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
WARMUP_RUNS = 1
MEASURED_RUNS = 2


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_cached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate greedily while reusing attention key/value states."""
    generated = []

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        past_key_values = outputs.past_key_values
        generated.append(next_token)

        for _ in range(1, max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            past_key_values = outputs.past_key_values
            generated.append(next_token)

    return torch.cat(generated, dim=1)


def generate_uncached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate greedily by recomputing the entire sequence each step."""
    sequence = input_ids
    generated = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=sequence, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            generated.append(next_token)
            sequence = torch.cat([sequence, next_token], dim=1)

    return torch.cat(generated, dim=1)


def benchmark(
    name: str,
    generate_fn,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    # Warm up the execution path; do not include this in the measurements.
    for _ in range(WARMUP_RUNS):
        generate_fn(model, input_ids, MAX_NEW_TOKENS)

    latencies = []
    reference_output = None

    for _ in range(MEASURED_RUNS):
        synchronize_if_needed(device)
        start = time.perf_counter()

        output = generate_fn(model, input_ids, MAX_NEW_TOKENS)

        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        if reference_output is None:
            reference_output = output
        elif not torch.equal(reference_output, output):
            raise RuntimeError(f"{name} produced inconsistent token IDs")

    median_latency = statistics.median(latencies)
    print(f"\n{name}")
    print(f"  Runs: {latencies}")
    print(f"  Median latency: {median_latency:.4f} s")
    print(f"  Generated tokens: {reference_output.shape[1]}")
    print(f"  Throughput: {MAX_NEW_TOKENS / median_latency:.2f} tokens/s")

    return reference_output, latencies


def main() -> None:
    generator = PretrainedGenerator(model_id=MODEL_ID, device="cpu")
    model = generator.model
    device = generator.device

    encoded = generator.tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    print("=== KV Cache Comparison ===")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Target generated tokens: {MAX_NEW_TOKENS}")
    print(f"Measured runs per mode: {MEASURED_RUNS}")

    cached_output, cached_times = benchmark(
        "Cached decoding",
        generate_cached,
        model,
        input_ids,
        device,
    )
    uncached_output, uncached_times = benchmark(
        "Uncached decoding",
        generate_uncached,
        model,
        input_ids,
        device,
    )

    same_output = torch.equal(cached_output, uncached_output)
    cached_median = statistics.median(cached_times)
    uncached_median = statistics.median(uncached_times)
    speedup = uncached_median / cached_median

    print("\n=== Comparison ===")
    print(f"Token IDs match: {same_output}")
    print(f"Uncached / cached latency ratio: {speedup:.2f}x")
    print(
        "Latency reduction: "
        f"{(1 - cached_median / uncached_median) * 100:.1f}%"
    )

    print("\n=== Cached Output ===")
    print(
        generator.tokenizer.decode(
            torch.cat([input_ids, cached_output], dim=1)[0],
            skip_special_tokens=True,
        )
    )

    if not same_output:
        raise RuntimeError(
            "Cached and uncached decoding differ. Investigate before "
            "interpreting the benchmark."
        )


if __name__ == "__main__":
    main()