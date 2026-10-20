"""Filesystem layout. Datasets and artifacts live outside the repo (default ~/data)."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("LOOPBENCH_DATA", Path.home() / "data"))
WORK_ROOT = DATA_ROOT / "loopbench"
ARTIFACTS = WORK_ROOT / "artifacts"
LEDGER = WORK_ROOT / "ledger"


def dataset_dir(name: str) -> Path:
    return DATA_ROOT / "loopbench-datasets" / name


def artifact_dir(plan_hash: str) -> Path:
    return ARTIFACTS / plan_hash[:2] / plan_hash
