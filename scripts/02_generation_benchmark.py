import statistics
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "gpt2"
PROMPT = "A production language model"
OUTPUT_LENGTHS = [16,40,80]
REPEATS = 3

def main()->None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.inference_mode():

        model.generate(
            **inputs,
            min_new_tokens=4,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id = tokenizer.eos_token_id,
        )

    print("\n--Generation benchmark ---")

    for token_count in OUTPUT_LENGTHS:
        latencies=[]
        actual_token_counts = []

        for run_number in range(1,REPEATS+1):
            start = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    min_new_tokens = token_count,
                    max_new_tokens = token_count,
                    do_sample =False,
                    pad_token_id = tokenizer.eos_token_id,
                )

            elapsed = time.perf_counter() - start
            generated = output.shape[1] - inputs["input_ids"].shape[1]

            latencies.append(elapsed)
            actual_token_counts.append(generated)

            print(
                f"Tokens = {token_count:3} | Run = {run_number}"
                f"| Latency = {elapsed:.3f}s | Actual tokens = {generated}"
            )
        median_latency = statistics.median(latencies)

        median_throughput = token_count / median_latency

        print(
            f"SUMMARY | Tokens={token_count:3} "
            f"| Median latency={median_latency:.3f}s "
            f"| Throughput={median_throughput:.2f} tokens/sec"
        )
        print()


if __name__ == "__main__":
    main()
