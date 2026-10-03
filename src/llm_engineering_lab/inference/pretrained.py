
import time
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationResult:
    """Output and performance measurements for one generation call."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


class PretrainedGenerator:
    """Reusable inference wrapper around a Hugging Face causal LM."""

    def __init__(
        self,
        model_id: str = "gpt2",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)

        load_start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.load_seconds = time.perf_counter() - load_start

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        do_sample: bool = False,
    ) -> GenerationResult:
        """Generate text and measure end-to-end generation-call latency."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }
        prompt_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        latency_seconds = time.perf_counter() - start
        generated_tokens = output.shape[1] - prompt_tokens

        text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        throughput = (
            generated_tokens / latency_seconds
            if latency_seconds > 0
            else 0.0
        )

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_seconds=latency_seconds,
            tokens_per_second=throughput,
        )