
import torch

from llm_engineering_lab.inference.pretrained import PretrainedGenerator


MODEL_ID = "gpt2"
SEQUENCE_LENGTHS = [16, 64, 256, 512]
BATCH_SIZES = [1, 2]


def theoretical_cache_bytes(
    num_layers: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    bytes_per_element: int,
) -> int:
    return (
        2
        * num_layers
        * batch_size
        * sequence_length
        * hidden_size
        * bytes_per_element
    )


def actual_cache_bytes(cache) -> int | None:
    """Count key/value tensor storage for common Transformers cache formats."""
    total = 0
    found = False

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        tensors = list(cache.key_cache) + list(cache.value_cache)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
                found = True
        return total if found else None

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for attribute in ("keys", "values"):
                tensor = getattr(layer, attribute, None)
                if isinstance(tensor, torch.Tensor):
                    total += tensor.numel() * tensor.element_size()
                    found = True
        return total if found else None

    if isinstance(cache, (tuple, list)):
        for item in cache:
            size = actual_cache_bytes(item)
            if size is not None:
                total += size
                found = True
        return total if found else None

    return None


def mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def main() -> None:
    generator = PretrainedGenerator(model_id=MODEL_ID, device="cpu")
    model = generator.model
    tokenizer = generator.tokenizer
    device = generator.device
    config = model.config

    num_layers = config.n_layer
    hidden_size = config.n_embd
    bytes_per_element = next(model.parameters()).element_size()

    prompt = (
        "Efficient language model serving requires careful management "
        "of memory, compute, concurrency, and latency. "
    ) * 100
    
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(SEQUENCE_LENGTHS),
    )
    
    all_ids = encoded["input_ids"].to(device)
    
    

    print("=== KV Cache Memory Scaling ===")
    print(f"Model: {MODEL_ID}")
    print(f"Layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Cache element size: {bytes_per_element} bytes")
    print()
    print(
        f"{'Batch':>6} {'Seq len':>8} {'Estimated MiB':>15} "
        f"{'Observed MiB':>14} {'Difference':>12}"
    )

    for batch_size in BATCH_SIZES:
        for sequence_length in SEQUENCE_LENGTHS:
            if sequence_length > all_ids.shape[1]:
                raise ValueError("Prompt is shorter than a requested sequence")

            input_ids = all_ids[:, :sequence_length].repeat(batch_size, 1)

            with torch.inference_mode():
                outputs = model(input_ids=input_ids, use_cache=True)

            estimated = theoretical_cache_bytes(
                num_layers=num_layers,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                bytes_per_element=bytes_per_element,
            )
            observed = actual_cache_bytes(outputs.past_key_values)

            if observed is None:
                observed_text = "unsupported format"
                difference_text = "n/a"
            else:
                observed_text = f"{mib(observed):.3f}"
                difference_text = f"{mib(observed - estimated):+.3f}"

            print(
                f"{batch_size:>6} {sequence_length:>8} "
                f"{mib(estimated):>15.3f} {observed_text:>14} "
                f"{difference_text:>12}"
            )

            del outputs

    print("\nNote: estimates cover KV tensors, not model weights or activations.")


if __name__ == "__main__":
    main()