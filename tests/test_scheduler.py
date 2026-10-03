
from llm_engineering_lab.inference.batch import WorkType
from llm_engineering_lab.inference.request import Request, RequestState
from llm_engineering_lab.inference.scheduler import Scheduler, SchedulerConfig


def make_request(
    request_id: str,
    prompt_tokens: int = 50,
    output_tokens: int = 10,
    priority: int = 1,
    slo_seconds: float = 2.0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        expected_output_tokens=output_tokens,
        priority=priority,
        slo_seconds=slo_seconds,
    )


def test_empty_scheduler_builds_empty_batch():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))

    batch = scheduler.build_batch()

    assert batch.items == []
    assert batch.total_tokens == 0


def test_scheduler_respects_prompt_token_budget():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    first = make_request("first", prompt_tokens=60)
    second = make_request("second", prompt_tokens=60)

    scheduler.add_request(first)
    scheduler.add_request(second)

    selected = scheduler.schedule_waiting()

    assert len(selected) == 1
    assert selected[0].request_id == "first"


def test_scheduler_skips_oversized_request():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    oversized = make_request("oversized", prompt_tokens=150)
    small = make_request("small", prompt_tokens=20)

    scheduler.add_request(oversized)
    scheduler.add_request(small)

    selected = scheduler.schedule_waiting()

    assert [request.request_id for request in selected] == ["small"]


def test_scheduler_rejects_request_that_misses_service_time_estimate():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    infeasible = make_request(
        "infeasible",
        output_tokens=100,
        slo_seconds=0.5,
    )

    scheduler.add_request(infeasible)

    assert scheduler.schedule_waiting() == []


def test_scheduler_prefers_lower_remaining_slo_slack():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=200))

    relaxed = make_request("relaxed", slo_seconds=2.0)
    urgent = make_request("urgent", slo_seconds=0.5)

    scheduler.add_request(relaxed)
    scheduler.add_request(urgent)

    selected = scheduler.schedule_waiting()

    assert [request.request_id for request in selected] == [
        "urgent",
        "relaxed",
    ]


def test_admission_moves_request_to_running():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    request = make_request("admit")

    scheduler.add_request(request)
    scheduler.admit_requests([request])

    assert request.state == RequestState.RUNNING
    assert request not in scheduler.waiting_queue
    assert request in scheduler.running_requests


def test_mark_finished_removes_running_request():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=100))
    request = make_request("finish")

    scheduler.add_request(request)
    scheduler.admit_requests([request])
    scheduler.mark_finished(request)

    assert request.state == RequestState.FINISHED
    assert request not in scheduler.running_requests
    assert request in scheduler.finished_requests


def test_build_batch_separates_prefill_and_decode():
    scheduler = Scheduler(SchedulerConfig(max_batch_tokens=200))
    waiting = make_request("waiting", prompt_tokens=50)
    running = make_request("running", prompt_tokens=30)

    scheduler.add_request(waiting)
    scheduler.add_request(running)
    scheduler.admit_requests([running])

    batch = scheduler.build_batch()

    assert batch.prefill_tokens == 50
    assert batch.decode_tokens == 1
    assert [item.work_type for item in batch.items] == [
        WorkType.PREFILL,
        WorkType.DECODE,
    ]