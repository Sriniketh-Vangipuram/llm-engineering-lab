
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean


# Approximate seconds per 24-token generation run from the
# earlier CPU benchmark. These values are illustrative, not universal.
BATCH_STEP_SECONDS = {
    1: 1.585 / 24,
    2: 1.843 / 24,
    4: 2.218 / 24,
    8: 2.401 / 24,
}

MAX_BATCH_SIZE = 8
MAX_BATCH_WAIT_SECONDS = 0.15


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time: float
    output_tokens: int


@dataclass
class RequestResult:
    request_id: int
    arrival_time: float
    generation_start_time: float
    first_token_time: float
    completion_time: float
    output_tokens: int

    @property
    def queue_wait(self) -> float:
        """Time between arrival and admission for generation."""
        return self.generation_start_time - self.arrival_time

    @property
    def time_to_first_token(self) -> float:
        """Arrival-to-first-token latency, including queueing."""
        return self.first_token_time - self.arrival_time

    @property
    def generation_time(self) -> float:
        """Time from generation start to completion."""
        return self.completion_time - self.generation_start_time

    @property
    def latency(self) -> float:
        """Total arrival-to-completion latency."""
        return self.completion_time - self.arrival_time


def step_seconds(batch_size: int) -> float:
    """Linearly interpolate the measured timing anchors."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    anchors = sorted(BATCH_STEP_SECONDS.items())

    if batch_size <= anchors[0][0]:
        return anchors[0][1]

    if batch_size >= anchors[-1][0]:
        return anchors[-1][1]

    for (size_a, time_a), (size_b, time_b) in zip(
        anchors, anchors[1:]
    ):
        if size_a <= batch_size <= size_b:
            ratio = (batch_size - size_a) / (size_b - size_a)
            return time_a + ratio * (time_b - time_a)

    raise ValueError(f"Unsupported batch size: {batch_size}")


def make_workload() -> list[Request]:
    """Create deterministic requests with staggered arrivals."""
    arrivals = [
        0.00, 0.02, 0.04, 0.06, 0.08,
        0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.22, 0.24, 0.26, 0.28,
        0.30, 0.32, 0.34, 0.36, 0.38,
        0.40, 0.42, 0.44, 0.46,
    ]

    output_lengths = [
        8, 32, 12, 48, 16, 8, 40, 24,
        12, 56, 16, 8, 32, 20, 48, 12,
        8, 40, 24, 16, 56, 8, 32, 12,
    ]

    return [
        Request(request_id=i, arrival_time=arrival, output_tokens=tokens)
        for i, (arrival, tokens) in enumerate(
            zip(arrivals, output_lengths), start=1
        )
    ]


def simulate_static_batching(
    requests: list[Request],
) -> list[RequestResult]:
    """
    Build fixed batches and wait for the longest request in each batch.

    Requests that arrive after a batch starts cannot join that batch.
    Finished short requests leave their slots idle until the batch ends.
    """
    pending = sorted(requests, key=lambda request: request.arrival_time)
    results: list[RequestResult] = []
    current_time = 0.0

    while pending:
        available = [
            request
            for request in pending
            if request.arrival_time <= current_time
        ]

        if not available:
            current_time = pending[0].arrival_time
            available = [
                request
                for request in pending
                if request.arrival_time <= current_time
            ]

        first_arrival = available[0].arrival_time
        deadline = first_arrival + MAX_BATCH_WAIT_SECONDS

        while len(available) < MAX_BATCH_SIZE:
            future = [
                request
                for request in pending
                if request.arrival_time > current_time
            ]

            if not future:
                break

            next_arrival = future[0].arrival_time

            if next_arrival > deadline:
                current_time = deadline
                break

            current_time = next_arrival
            available = [
                request
                for request in pending
                if request.arrival_time <= current_time
            ]

        batch = available[:MAX_BATCH_SIZE]

        if not batch:
            continue

        batch_start = current_time
        per_token_time = step_seconds(len(batch))

        for request in batch:
            first_token_time = batch_start + per_token_time
            completion_time = (
                batch_start + request.output_tokens * per_token_time
            )

            results.append(
                RequestResult(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    generation_start_time=batch_start,
                    first_token_time=first_token_time,
                    completion_time=completion_time,
                    output_tokens=request.output_tokens,
                )
            )

        # Static batching keeps the batch occupied until its longest
        # request finishes, even if shorter requests finish earlier.
        current_time = batch_start + (
            max(request.output_tokens for request in batch)
            * per_token_time
        )

        batch_ids = {request.request_id for request in batch}
        pending = [
            request
            for request in pending
            if request.request_id not in batch_ids
        ]

    return results


def simulate_continuous_admission(
    requests: list[Request],
) -> list[RequestResult]:
    """
    Admit requests into free slots after each decode step.

    Each active request generates one token per step. When a request
    finishes, a waiting request can enter the next decode step.
    """
    pending = sorted(requests, key=lambda request: request.arrival_time)
    active: dict[int, dict] = {}
    results: list[RequestResult] = []
    current_time = 0.0

    while pending or active:
        # Admit arrived requests while capacity remains.
        while (
            pending
            and pending[0].arrival_time <= current_time
            and len(active) < MAX_BATCH_SIZE
        ):
            request = pending.pop(0)
            active[request.request_id] = {
                "request": request,
                "generated": 0,
                "generation_start_time": current_time,
                "first_token_time": None,
            }

        if not active:
            current_time = pending[0].arrival_time
            continue

        step_start = current_time
        per_token_time = step_seconds(len(active))
        step_end = step_start + per_token_time

        for state in active.values():
            state["generated"] += 1

            if state["first_token_time"] is None:
                state["first_token_time"] = step_end

        finished_ids = [
            request_id
            for request_id, state in active.items()
            if state["generated"] >= state["request"].output_tokens
        ]

        for request_id in finished_ids:
            state = active.pop(request_id)
            request = state["request"]

            results.append(
                RequestResult(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    generation_start_time=state["generation_start_time"],
                    first_token_time=state["first_token_time"],
                    completion_time=step_end,
                    output_tokens=request.output_tokens,
                )
            )

        current_time = step_end

    return results


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, where p is between 0 and 1."""
    if not values:
        raise ValueError("Cannot calculate percentile of empty values")

    if not 0 < p <= 1:
        raise ValueError("p must be in the interval (0, 1]")

    ordered = sorted(values)
    index = max(0, ceil(p * len(ordered)) - 1)
    return ordered[index]


def report(
    name: str,
    requests: list[Request],
    results: list[RequestResult],
) -> None:
    if not results:
        print(f"\n{name}: no completed requests")
        return

    by_id = {result.request_id: result for result in results}
    ordered = [by_id[request.request_id] for request in requests]

    total_tokens = sum(result.output_tokens for result in ordered)
    first_arrival = min(request.arrival_time for request in requests)
    last_completion = max(result.completion_time for result in ordered)
    makespan = last_completion - first_arrival

    if makespan <= 0:
        raise ValueError("Workload makespan must be positive")

    queue_waits = [result.queue_wait for result in ordered]
    ttfts = [result.time_to_first_token for result in ordered]
    latencies = [result.latency for result in ordered]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Completed requests:       {len(results)}")
    print(f"Total generated tokens:   {total_tokens}")
    print(f"Workload makespan:        {makespan:.3f} s")
    print(f"Request throughput:       {len(results) / makespan:.2f} req/s")
    print(f"Token throughput:         {total_tokens / makespan:.2f} tokens/s")
    print(f"Mean queue wait:          {mean(queue_waits):.3f} s")
    print(f"Mean time to first token: {mean(ttfts):.3f} s")
    print(f"P95 time to first token:  {percentile(ttfts, 0.95):.3f} s")
    print(f"Mean request latency:     {mean(latencies):.3f} s")
    print(f"P95 request latency:      {percentile(latencies, 0.95):.3f} s")


def main() -> None:
    requests = make_workload()

    static_results = simulate_static_batching(requests)
    continuous_results = simulate_continuous_admission(requests)

    report("Static batching", requests, static_results)
    report("Continuous admission", requests, continuous_results)


if __name__ == "__main__":
    main()