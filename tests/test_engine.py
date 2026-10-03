
from llm_engineering_lab.inference.engine import InferenceEngine
from llm_engineering_lab.inference.request import Request, RequestState
from llm_engineering_lab.inference.scheduler import Scheduler, SchedulerConfig


def test_engine_prefills_then_decodes_and_finishes_request():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    engine = InferenceEngine(scheduler)

    request = Request(
        request_id="engine-test",
        prompt_tokens=10,
        expected_output_tokens=1,
        priority=1,
        slo_seconds=1.0,
    )
    scheduler.add_request(request)

    prefill_batch = engine.run_step()

    assert prefill_batch is not None
    assert prefill_batch.prefill_tokens == 10
    assert prefill_batch.decode_tokens == 0
    assert request.state == RequestState.RUNNING
    assert request.generated_tokens == 0

    decode_batch = engine.run_step()

    assert decode_batch is not None
    assert decode_batch.prefill_tokens == 0
    assert decode_batch.decode_tokens == 1
    assert request.generated_tokens == 1
    assert request.state == RequestState.FINISHED
    assert request in scheduler.finished_requests

    assert engine.run_step() is None