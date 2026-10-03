
import json

import yaml

from llm_engineering_lab.config import ExperimentConfig
from llm_engineering_lab.experiments.runner import initialize_run


def test_initialize_run_creates_expected_artifacts(tmp_path):
    config = ExperimentConfig(
        experiment_id="test-experiment",
        description="Test experiment initialization",
        output_dir=tmp_path / "artifacts",
    )

    run_dir = initialize_run(config)

    assert run_dir.is_dir()
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "logs" / "run.log").is_file()

    saved_config = yaml.safe_load(
        (run_dir / "config.yaml").read_text(encoding="utf-8")
    )
    assert saved_config["experiment_id"] == "test-experiment"
    assert saved_config["seed"] == 42

    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics == {}

    environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["python_version"]
    assert "package_versions" in environment

    metadata = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "initialized"
    assert metadata["experiment_id"] == "test-experiment"
    assert metadata["started_at"] is None
    assert metadata["completed_at"] is None


def test_each_run_gets_a_unique_directory(tmp_path):
    config = ExperimentConfig(
        experiment_id="repeatable-test",
        description="Test unique run IDs",
        output_dir=tmp_path / "artifacts",
    )

    first_run = initialize_run(config)
    second_run = initialize_run(config)

    assert first_run != second_run
    assert first_run.is_dir()
    assert second_run.is_dir()