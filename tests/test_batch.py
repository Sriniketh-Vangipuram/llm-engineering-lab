
from llm_engineering_lab.inference.batch import Batch, BatchItem, WorkType
from llm_engineering_lab.inference.request import Request


def make_request(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_tokens=100,
        expected_output_tokens=20,
        priority=1,
        slo_seconds=5.0,
    )


def test_batch_token_accounting():
    prefill_request = make_request("prefill")
    decode_request = make_request("decode")

    batch = Batch(
        items=[
            BatchItem(
                request=prefill_request,
                work_type=WorkType.PREFILL,
                tokens=100,
            ),
            BatchItem(
                request=decode_request,
                work_type=WorkType.DECODE,
                tokens=1,
            ),
        ]
    )

    assert batch.total_tokens == 101
    assert batch.prefill_tokens == 100
    assert batch.decode_tokens == 1


def test_empty_batch_has_zero_tokens():
    batch = Batch(items=[])

    assert batch.total_tokens == 0
    assert batch.prefill_tokens == 0
    assert batch.decode_tokens == 0