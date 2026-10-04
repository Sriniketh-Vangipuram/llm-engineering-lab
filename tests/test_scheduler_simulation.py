
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "10_scheduler_simulation.py"
)

SPEC = importlib.util.spec_from_file_location(
    "scheduler_simulation",
    SCRIPT_PATH,
)

assert SPEC is not None
assert SPEC.loader is not None

simulation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = simulation
SPEC.loader.exec_module(simulation)


@pytest.mark.parametrize(
    "simulate",
    [
        simulation.simulate_static_batching,
        simulation.simulate_continuous_admission,
    ],
)
def test_every_request_completes_exactly_once(simulate):
    requests = simulation.make_workload()

    results = simulate(requests)

    result_ids = [result.request_id for result in results]
    expected_ids = [request.request_id for request in requests]

    assert len(results) == len(requests)
    assert sorted(result_ids) == sorted(expected_ids)


@pytest.mark.parametrize(
    "simulate",
    [
        simulation.simulate_static_batching,
        simulation.simulate_continuous_admission,
    ],
)
def test_timing_metrics_are_consistent(simulate):
    requests = simulation.make_workload()
    results = simulate(requests)

    for result in results:
        assert result.arrival_time <= result.generation_start_time
        assert result.generation_start_time < result.first_token_time
        assert result.first_token_time <= result.completion_time

        assert result.queue_wait == pytest.approx(
            result.generation_start_time - result.arrival_time
        )
        assert result.time_to_first_token == pytest.approx(
            result.first_token_time - result.arrival_time
        )
        assert result.latency == pytest.approx(
            result.completion_time - result.arrival_time
        )


@pytest.mark.parametrize(
    "simulate",
    [
        simulation.simulate_static_batching,
        simulation.simulate_continuous_admission,
    ],
)
def test_generated_token_accounting_is_preserved(simulate):
    requests = simulation.make_workload()
    results = simulate(requests)

    expected_tokens = sum(request.output_tokens for request in requests)
    actual_tokens = sum(result.output_tokens for result in results)

    assert actual_tokens == expected_tokens


def test_static_batching_finishes_longer_request_later():
    requests = [
        simulation.Request(1, 0.0, 2),
        simulation.Request(2, 0.0, 6),
    ]

    results = simulation.simulate_static_batching(requests)
    by_id = {result.request_id: result for result in results}

    assert (
        by_id[1].generation_start_time
        == by_id[2].generation_start_time
    )
    assert by_id[1].completion_time < by_id[2].completion_time


def test_step_seconds_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        simulation.step_seconds(0)


@pytest.mark.parametrize("p", [0, -0.1, 1.1])
def test_percentile_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        simulation.percentile([1.0, 2.0, 3.0], p)


def test_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        simulation.percentile([], 0.95)