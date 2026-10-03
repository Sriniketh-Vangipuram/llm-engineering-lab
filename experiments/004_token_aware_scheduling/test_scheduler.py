from llm_engineering_lab.inference.request import Request
from llm_engineering_lab.inference.scheduler import (
    Scheduler,
    SchedulerConfig,
)
from llm_engineering_lab.inference.engine import InferenceEngine


def main():
    scheduler = Scheduler(
        SchedulerConfig(max_batch_tokens=500)
    )

    # Waiting requests
    scheduler.add_request(
        Request(
            request_id="A",
            prompt_tokens=300,
            expected_output_tokens=20,
            priority=10,
            slo_seconds=5.0,
        )
    )

    scheduler.add_request(
        Request(
            request_id="B",
            prompt_tokens=100,
            expected_output_tokens=15,
            priority=3,
            slo_seconds=3.0,
        )
    )

    scheduler.add_request(
        Request(
            request_id="C",
            prompt_tokens=50,
            expected_output_tokens=10,
            priority=8,
            slo_seconds=5.0,
        )
    )

    engine = InferenceEngine(scheduler)

    # Run several inference iterations
    for _ in range(30):
        batch = engine.run_step()

        if batch is None:
            print("\nNo more work.")
            break

    print("\nFinal scheduler state:")

    print(
        "Waiting:",
        [r.request_id for r in scheduler.waiting_queue]
    )

    print(
    "Running:",
    [r.request_id for r in scheduler.running_requests]
    )

    print(
    "Finished:",
    [r.request_id for r in scheduler.finished_requests]
    )

if __name__ == "__main__":
    main()
