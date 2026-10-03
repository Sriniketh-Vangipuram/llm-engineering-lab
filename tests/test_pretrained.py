
import pytest
import torch

import llm_engineering_lab.inference.pretrained as pretrained


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "A test response"


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.eval_called = False

    def to(self, device):
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return iter([self.parameter])

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id):
        assert max_new_tokens == 3
        assert do_sample is False
        assert pad_token_id == 0
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_generator_loads_model_and_generates(monkeypatch):
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    monkeypatch.setattr(
        pretrained.AutoTokenizer,
        "from_pretrained",
        lambda model_id: fake_tokenizer,
    )
    monkeypatch.setattr(
        pretrained.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id: fake_model,
    )

    generator = pretrained.PretrainedGenerator(model_id="fake-model")
    result = generator.generate("Test prompt", max_new_tokens=3)

    assert generator.model_id == "fake-model"
    assert generator.load_seconds >= 0
    assert fake_model.eval_called
    assert result.text == "A test response"
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 3
    assert result.latency_seconds >= 0
    assert result.tokens_per_second >= 0


def test_generator_rejects_empty_prompt():
    generator = pretrained.PretrainedGenerator.__new__(
        pretrained.PretrainedGenerator
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generator.generate("   ")


def test_generator_rejects_invalid_token_budget():
    generator = pretrained.PretrainedGenerator.__new__(
        pretrained.PretrainedGenerator
    )

    with pytest.raises(ValueError, match="max_new_tokens"):
        generator.generate("hello", max_new_tokens=0)