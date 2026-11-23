import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = "benchmarks.run_v04_identifiability"


def run_cli(cwd, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-m", MODULE, *map(str, args)],
        cwd=cwd, env=env, text=True, capture_output=True, check=False,
    )


def check_result(payload):
    assert {"benchmark", "date", "environment", "result"} <= payload.keys()
    result = payload["result"]
    assert result["all_rigid"] is True
    case = result["left_zero_counterexample"]
    assert case["size"] == 3
    assert case["blind_regret"] == pytest.approx(2 / 3)
    assert case["informed_regret"] == pytest.approx(2 / 3)
    assert case["gap"] == pytest.approx(0)


def test_stdout_from_repository_root():
    ecc_existed = (ROOT / ".ecc").exists()
    result = run_cli(ROOT)
    assert result.returncode == 0, result.stderr
    check_result(json.loads(result.stdout))
    assert (ROOT / ".ecc").exists() == ecc_existed


def test_stdout_creates_no_files(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    check_result(json.loads(result.stdout))
    assert list(tmp_path.iterdir()) == []


def test_explicit_output_only(tmp_path):
    output = tmp_path / "отчёты" / "результат.json"
    result = run_cli(tmp_path, "--output", output)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    check_result(json.loads(output.read_text(encoding="utf-8")))
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()) == [output.relative_to(tmp_path)]


@pytest.mark.parametrize("argument", ["--unknown", "--output"])
def test_invalid_arguments_fail_without_files(tmp_path, argument):
    result = run_cli(tmp_path, argument)
    assert result.returncode != 0
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []
