#!/usr/bin/env python3
"""Regression checks for the Mellum oracle comparator's failure modes."""

import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tests" / "check_mellum_layer0_oracle.py"
REFERENCE = ROOT / "tests" / "test-vectors" / "mellum-llama-cpp" / "l-out-27-tokenwise.f32"
LOGITS_REFERENCE = ROOT / "tests" / "test-vectors" / "mellum-llama-cpp" / "result-output-tokenwise.f32"


def run_checker(
    actual: Path, cwd: Path, layer: str = "27-tokenwise"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--layer", layer, str(actual)],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_rejected(value: float, label: str, scratch: Path) -> None:
    actual = scratch / f"{label}.f32"
    shutil.copyfile(REFERENCE, actual)
    with actual.open("r+b") as file:
        file.write(struct.pack("<f", value))
    result = run_checker(actual, scratch)
    if result.returncode == 0 or "non-finite actual value at index 0" not in result.stderr:
        raise AssertionError(f"{label} output was not rejected: {result.stderr!r}")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        actual = scratch / "valid.f32"
        shutil.copyfile(REFERENCE, actual)
        result = run_checker(actual, scratch)
        if result.returncode != 0:
            raise AssertionError(f"valid oracle failed outside the repository: {result.stderr!r}")
        logits = scratch / "valid-logits.f32"
        shutil.copyfile(LOGITS_REFERENCE, logits)
        result = run_checker(logits, scratch, "logits-tokenwise")
        if result.returncode != 0:
            raise AssertionError(f"valid logits oracle failed outside the repository: {result.stderr!r}")
        assert_rejected(math.nan, "nan", scratch)
        assert_rejected(math.inf, "inf", scratch)
    print("Mellum oracle checker: valid, NaN, and Inf cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
