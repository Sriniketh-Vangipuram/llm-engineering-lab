
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "gpt2"
PROMPT = "A production language model"
OUTPUT_TOKENS = 40
THREAD_COUNTS = [1, 2, 4]
WARMUP_RUNS = 1
MEASURED_RUNS = 7
ARTIFACT_DIR = Path("artifacts/benchmarks")


def generate_once(model, tokenizer, inputs, token_count):
    """Generate a fixed number of tokens and return elapsed seconds."""
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            min_new_tokens=token_count,
            max_new_tokens=token_count,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start
    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    if generated_tokens != token_count:
        raise RuntimeError(
            f"Expected {token_count} tokens, generated {generated_tokens}"
        )

    return elapsed


def summarize_thread_count(
    model, tokenizer, inputs, thread_count
) -> dict:
    """Warm up and measure generation at one CPU thread setting."""
    torch.set_num_threads(thread_count)

    for _ in range(WARMUP_RUNS):
        generate_once(model, tokenizer, inputs, OUTPUT_TOKENS)

    latencies = [
        generate_once(model, tokenizer, inputs, OUTPUT_TOKENS)
        for _ in range(MEASURED_RUNS)
    ]

    median_latency = statistics.median(latencies)

    return {
        "threads": torch.get_num_threads(),
        "output_tokens": OUTPUT_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "latency_seconds": [round(value, 6) for value in latencies],
        "median_latency_seconds": round(median_latency, 6),
        "mean_latency_seconds": round(statistics.mean(latencies), 6),
        "min_latency_seconds": round(min(latencies), 6),
        "max_latency_seconds": round(max(latencies), 6),
        "latency_stdev_seconds": round(
            statistics.stdev(latencies), 6
        ),
        "median_throughput_tokens_per_second": round(
            OUTPUT_TOKENS / median_latency, 3
        ),
    }


def main() -> None:
    print(f"Loading {MODEL_ID}...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    model_load_seconds = time.perf_counter() - load_start
    inputs = tokenizer(PROMPT, return_tensors="pt")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    results = []
    for thread_count in THREAD_COUNTS:
        print(f"\nBenchmarking with {thread_count} thread(s)...")
        result = summarize_thread_count(
            model, tokenizer, inputs, thread_count
        )
        results.append(result)

        print(
            f"Median latency: {result['median_latency_seconds']:.3f}s | "
            f"Throughput: "
            f"{result['median_throughput_tokens_per_second']:.2f} tokens/sec"
        )

    report = {
        "experiment": "cpu-thread-generation-benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "parameter_count": parameter_count,
        "device": str(next(model.parameters()).device),
        "model_load_seconds": round(model_load_seconds, 3),
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "results": results,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ARTIFACT_DIR / f"cpu_threads_{timestamp}.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved benchmark report: {report_path}")


if __name__ == "__main__":
    main()