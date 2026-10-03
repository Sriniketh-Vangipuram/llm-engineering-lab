
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from llm_engineering_lab.config import ExperimentConfig


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _package_versions() -> dict[str, str]:
    package_names = (
        "llm-engineering-lab",
        "torch",
        "transformers",
        "pydantic",
        "PyYAML",
    )
    versions: dict[str, str] = {}

    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            continue

    return versions


def _git_revision() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.strip()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def initialize_run(config: ExperimentConfig) -> Path:
    """Create an experiment run directory and initial metadata.

    This initializes a run; it does not execute a model or experiment.
    """

    timestamp = _utc_now()
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "logs").mkdir()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "logs" / "run.log").write_text(
        "Run initialized. Experiment execution has not started.\n",
        encoding="utf-8",
    )

    environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": _package_versions(),
        "git_revision": _git_revision(),
    }
    _write_json(run_dir / "environment.json", environment)

    metadata = {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "status": "initialized",
        "created_at": timestamp.isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    _write_json(run_dir / "run.json", metadata)

    return run_dir