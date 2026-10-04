import statistics
import time

import torch

from llm_engineering_lab.inference.pretrained import PretrainedGenerator

MODEL_ID = "gpt2"
PROMPT_LENGTHS = [16, 64]
OUTPUT_LENGTHS = [16, 40]
REPEATS = 3

PROMPT_TEXT = (
    "Production language model inference requires careful management "
    "of computation, memory, latency, throughput, batching, and "
    "reliability across many concurrent user requests. "
) * 30


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def generate_cached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
)->torch.Tensor:
    
    generated = []
    
    with torch.inference_mode():
        
        outputs = model(input_ids = input_ids, use_cache = True)
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        
        past_key_values = outputs.past_key_values
        generated.append(next_token)
        
        for _ in range(1, max_new_tokens):
            outputs = model(
                input_ids = next_token,
                past_key_values = past_key_values,
                use_cache = True,
            )
            next_token = outputs.logits[:,-1,:].argmax(
                dim=-1,keepdim=True
            )
            past_key_values = outputs.past_key_values
            generated.append(next_token)
            
    return torch.cat(generated,dim=1)

def generate_uncached(
    model:torch.nn.Module,
    input_ids:torch.Tensor,
    max_new_tokens: int
)->torch.Tensor:
    
    sequence = input_ids
    generated=[]
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=sequence, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            generated.append(next_token)
            sequence = torch.cat([sequence, next_token], dim=1)

    return torch.cat(generated, dim=1)

def measure(
    generate_fn,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    output_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    # Warm-up is excluded from reported timings.
    with torch.inference_mode():
        generate_fn(model, input_ids, output_length)

    timings = []
    reference = None

    for _ in range(REPEATS):
        synchronize_if_needed(device)
        start = time.perf_counter()

        output = generate_fn(model, input_ids, output_length)

        synchronize_if_needed(device)
        timings.append(time.perf_counter() - start)

        if reference is None:
            reference = output
        elif not torch.equal(reference, output):
            raise RuntimeError("Output token IDs changed between repetitions")

    return reference, timings


def main() -> None:
    generator = PretrainedGenerator(model_id=MODEL_ID, device="cpu")
    model = generator.model
    device = generator.device

    encoded = generator.tokenizer(PROMPT_TEXT, return_tensors="pt")
    all_prompt_ids = encoded["input_ids"].to(device)

    if all_prompt_ids.shape[1] < max(PROMPT_LENGTHS):
        raise ValueError("Prompt text does not contain enough tokens")

    print("=== KV Cache Scaling Experiment ===")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Measured repetitions per configuration: {REPEATS}")
    print()
    print(
        f"{'Prompt':>8} {'Output':>8} {'Cached(s)':>12} "
        f"{'Uncached(s)':>13} {'Speedup':>10} {'Match':>8}"
    )

    for prompt_length in PROMPT_LENGTHS:
        input_ids = all_prompt_ids[:, :prompt_length]

        for output_length in OUTPUT_LENGTHS:
            cached_output, cached_times = measure(
                generate_cached,
                model,
                input_ids,
                output_length,
                device,
            )
            uncached_output, uncached_times = measure(
                generate_uncached,
                model,
                input_ids,
                output_length,
                device,
            )

            cached_median = statistics.median(cached_times)
            uncached_median = statistics.median(uncached_times)
            speedup = uncached_median / cached_median
            matches = torch.equal(cached_output, uncached_output)

            print(
                f"{prompt_length:>8} {output_length:>8} "
                f"{cached_median:>12.3f} {uncached_median:>13.3f} "
                f"{speedup:>9.2f}x {str(matches):>8}"
            )

            if not matches:
                raise RuntimeError(
                    f"Token mismatch for prompt={prompt_length}, "
                    f"output={output_length}"
                )

    print("\nAll configurations completed.")
    print("Times exclude model loading and tokenization.")


if __name__ == "__main__":
    main()