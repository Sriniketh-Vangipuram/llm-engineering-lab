import statistics
import time

import torch

from llm_engineering_lab.inference.pretrained import PretrainedGenerator


MODEL_ID = "gpt2"
PROMPT = (
    "A production inference server improves throughput by"
)
BATCH_SIZES = [1, 2, 4, 8]
MAX_NEW_TOKENS = 24
WARMUP_RUNS = 1
MEASURED_RUNS = 3


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
def generate_batch(
    model:torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
)->torch.Tensor:
    """Greedy decoding for a batch of equal-length prompts using KV cache."""
    
    generated = []
    
    with torch.inference_mode():
        outputs = model(input_ids = input_ids,use_cache = True)
        next_token = outputs.logits[:,-1,:].argmax(
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
            
    return torch.cat(generated, dim=1)

def median_latency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
)->tuple[float, torch.Tensor]:
    
    generate_batch(model, input_ids, MAX_NEW_TOKENS)
    
    latencies = []
    reference = None
    
    for _ in range(MEASURED_RUNS):
        synchronize_if_needed(device)
        start = time.perf_counter()
        
        output = generate_batch(model, input_ids, MAX_NEW_TOKENS)
        
        synchronize_if_needed(device)
        latencies.append(time.perf_counter() - start)
        
        if reference is None:
            reference = output
        elif not torch.equal(reference, output):
            raise RuntimeError("Output changed between repetitions")
        
    return statistics.median(latencies), reference

def main()->None:
    generator = PretrainedGenerator(model_id=MODEL_ID, device="cpu")
    model = generator.model
    tokenizer = generator.tokenizer
    device = generator.device

    encoded = tokenizer(PROMPT, return_tensors="pt")
    single_prompt = encoded["input_ids"].to(device)

    print("=== Batching Throughput Benchmark ===")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Prompt tokens: {single_prompt.shape[1]}")
    print(f"Output tokens per request: {MAX_NEW_TOKENS}")
    print(f"Measured repetitions: {MEASURED_RUNS}")
    print()
    print(
        f"{'Batch':>6} {'Batch time(s)':>15} "
        f"{'Requests/s':>12} {'Tokens/s':>12} "
        f"{'Seq time(s)':>13} {'Seq req/s':>11} {'Match':>8}"
    )

    for batch_size in BATCH_SIZES:
        batch_input = single_prompt.repeat(batch_size, 1)

        # Time processing all requests together.
        batch_time, batch_output = median_latency(
            model, batch_input, device
        )

        # Time the same requests one at a time.
        sequential_times = []
        sequential_outputs = []

        for _ in range(WARMUP_RUNS):
            generate_batch(model, single_prompt, MAX_NEW_TOKENS)

        for _ in range(MEASURED_RUNS):
            synchronize_if_needed(device)
            start = time.perf_counter()

            outputs = [
                generate_batch(model, single_prompt, MAX_NEW_TOKENS)
                for _ in range(batch_size)
            ]

            synchronize_if_needed(device)
            sequential_times.append(time.perf_counter() - start)
            sequential_outputs = outputs

        sequential_time = statistics.median(sequential_times)

        # Every repeated prompt should produce the same greedy output.
        match = all(
            torch.equal(batch_output[i:i + 1], output)
            for i, output in enumerate(sequential_outputs)
        )

        batch_requests_per_second = batch_size / batch_time
        batch_tokens_per_second = (
            batch_size * MAX_NEW_TOKENS / batch_time
        )
        sequential_requests_per_second = batch_size / sequential_time

        print(
            f"{batch_size:>6} {batch_time:>15.3f} "
            f"{batch_requests_per_second:>12.2f} "
            f"{batch_tokens_per_second:>12.2f} "
            f"{sequential_time:>13.3f} "
            f"{sequential_requests_per_second:>11.2f} "
            f"{str(match):>8}"
        )

        if not match:
            raise RuntimeError(
                f"Batch and sequential outputs differ at batch size {batch_size}"
            )

    print("\nAll configurations completed.")
    print("Times exclude model loading and tokenization.")
    print("Batch time is for the entire batch, not one individual request.")


if __name__ == "__main__":
    main()