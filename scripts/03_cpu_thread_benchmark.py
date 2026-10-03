
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "gpt2"
PROMPT = "A production language model"
OUTPUT_TOKENS = 40
REPEATS = 4
THREAD_COUNTS = [1, 2, 4]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")

    print(f"Default PyTorch threads: {torch.get_num_threads()}")
    print(f"Model: {MODEL_ID}")
    print(f"Output tokens per run: {OUTPUT_TOKENS}\n")

    for thread_count in THREAD_COUNTS:
        torch.set_num_threads(thread_count)

        # Warm up with the selected thread count.
        with torch.inference_mode():
            model.generate(
                **inputs,
                min_new_tokens=OUTPUT_TOKENS,
                max_new_tokens=OUTPUT_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        latencies = []

        for run_number in range(1, REPEATS + 1):
            start = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    min_new_tokens=OUTPUT_TOKENS,
                    max_new_tokens=OUTPUT_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            elapsed = time.perf_counter() - start
            generated = (
                output.shape[1] - inputs["input_ids"].shape[1]
            )
            latencies.append(elapsed)

            print(
                f"Threads={thread_count} | Run={run_number} "
                f"| Latency={elapsed:.3f}s | Tokens={generated}"
            )

        median_latency = statistics.median(latencies)
        throughput = OUTPUT_TOKENS / median_latency

        print(
            f"SUMMARY | Threads={thread_count} "
            f"| Median={median_latency:.3f}s "
            f"| Throughput={throughput:.2f} tokens/sec\n"
        )


if __name__ == "__main__":
    main()