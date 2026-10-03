from llm_engineering_lab.inference.request import (
    Request,
    RequestState,
)

from llm_engineering_lab.inference.scheduler import (
    Scheduler,
    SchedulerConfig,
)


scheduler = Scheduler(
    SchedulerConfig(
        max_batch_tokens=500
    )
)


A = Request(
    request_id="A",
    prompt_tokens=300,
    expected_output_tokens=800,
    priority=10,
    slo_seconds=2.0,
    queue_wait_seconds=1.2,
    generated_tokens=200,
    state=RequestState.RUNNING,
)

B = Request(
    request_id="B",
    prompt_tokens=100,
    expected_output_tokens=50,
    priority=3,
    slo_seconds=2.0,
    queue_wait_seconds=1.6,
    generated_tokens=35,
    state=RequestState.RUNNING,
)

C = Request(
    request_id="C",
    prompt_tokens=300,
    expected_output_tokens=100,
    priority=8,
    slo_seconds=2.0,
    queue_wait_seconds=1.4,
    state=RequestState.WAITING,
)

D = Request(
    request_id="D",
    prompt_tokens=100,
    expected_output_tokens=50,
    priority=2,
    slo_seconds=10.0,
    queue_wait_seconds=5.0,
    state=RequestState.WAITING,
)


for request in [A, B, C, D]:
    scheduler.add_request(request)


batch = scheduler.build_batch()

print(batch)

print("\nBatch items:")

for item in batch.items:
    print(
        f"{item.request.request_id}: "
        f"{item.work_type.value}, "
        f"{item.tokens} tokens"
    )

print("\nPrefill tokens:", batch.prefill_tokens)
print("Decode tokens:", batch.decode_tokens)
print("Total tokens:", batch.total_tokens)