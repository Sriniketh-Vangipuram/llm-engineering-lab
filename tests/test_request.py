
from llm_engineering_lab.inference.request import Request, RequestState


def test_request_initial_state_and_token_accounting():
    request = Request(
        request_id="req-1",
        prompt_tokens=100,
        expected_output_tokens=20,
        priority=1,
        slo_seconds=5.0,
    )

    assert request.state == RequestState.WAITING
    assert request.remaining_output_tokens == 20
    assert request.total_tokens == 120
    assert request.remaining_slo_slack == 5.0


def test_remaining_output_tokens_after_generation():
    request = Request(
        request_id="req-2",
        prompt_tokens=50,
        expected_output_tokens=10,
        priority=1,
        slo_seconds=2.0,
    )

    request.generated_tokens = 4

    assert request.remaining_output_tokens == 6


def test_remaining_output_tokens_never_go_below_zero():
    request = Request(
        request_id="req-3",
        prompt_tokens=10,
        expected_output_tokens=5,
        priority=1,
        slo_seconds=1.0,
    )

    request.generated_tokens = 7

    assert request.remaining_output_tokens == 0


def test_remaining_slo_slack_never_goes_below_zero():
    request = Request(
        request_id="req-4",
        prompt_tokens=10,
        expected_output_tokens=5,
        priority=1,
        slo_seconds=2.0,
    )

    request.queue_wait_seconds = 3.0

    assert request.remaining_slo_slack == 0