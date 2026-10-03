
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExperimentConfig(BaseModel):
    """Validated configuration shared by engineering experiments."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    description: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Path("artifacts")
    parameters: dict[str, Any] = Field(default_factory=dict)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration and validate its schema."""

    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping: {config_path}"
        )

    return ExperimentConfig.model_validate(raw_config)