from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_engineering_lab.config import (
    ExperimentConfig,
    load_experiment_config,
)


def test_load_valid_yaml_configuration(tmp_path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(
        """
experiment_id: test-run
description: Test configuration
seed: 123
output_dir: artifacts/test-run
parameters:
  batch_size: 8
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_file)

    assert config.experiment_id == "test-run"
    assert config.seed == 123
    assert config.parameters["batch_size"] == 8


def test_default_values_are_applied():
    config = ExperimentConfig(
        experiment_id="minimal",
        description="Minimal configuration",
    )

    assert config.seed == 42
    assert config.output_dir == Path("artifacts")
    assert config.parameters == {}


def test_invalid_seed_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="invalid",
            description="Invalid seed",
            seed=-1,
        )


def test_unknown_fields_are_rejected(tmp_path):
    config_file = tmp_path / "unknown.yaml"
    config_file.write_text(
        """
experiment_id: invalid
description: Unknown field test
unexpected_option: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_experiment_config(config_file)


def test_non_mapping_yaml_is_rejected(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("- item-one\n- item-two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_experiment_config(config_file)