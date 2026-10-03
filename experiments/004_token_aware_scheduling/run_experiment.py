from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    language: str
    tokens: int
    priority: int
    slo_seconds: float


REQUESTS = [
    Request("O1", "Odia", 40, 1, 20.0),
    Request("O2", "Odia", 40, 1, 20.0),
    Request("E1", "English", 10, 10, 2.0),
    Request("E2", "English", 10, 10, 2.0),
    Request("E3", "English", 10, 10, 2.0),
]

GPU_TOKEN_BUDGET = 100


def fifo_scheduler(requests, token_budget):
    batch = []
    used_tokens = 0

    for request in requests:
        if used_tokens + request.tokens <= token_budget:
            batch.append(request)
            used_tokens += request.tokens
        else:
            break

    return batch


def print_batch(name, batch):
    print("=" * 70)
    print(name)
    print("=" * 70)

    total_tokens = 0

    for request in batch:
        print(
            f"{request.request_id:>2} | "
            f"{request.language:<7} | "
            f"tokens={request.tokens:>2} | "
            f"priority={request.priority:>2} | "
            f"SLO={request.slo_seconds}s"
        )

        total_tokens += request.tokens

    print("-" * 70)
    print(f"Total tokens admitted: {total_tokens}")


if __name__ == "__main__":
    batch = fifo_scheduler(
        REQUESTS,
        GPU_TOKEN_BUDGET,
    )

    print_batch("FIFO SCHEDULER", batch)