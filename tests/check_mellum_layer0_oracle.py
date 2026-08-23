#!/usr/bin/env python3
"""Compare a ds4 Mellum diagnostic output with a pinned llama.cpp oracle."""

import argparse
import hashlib
import math
import struct
import sys
from pathlib import Path


HIDDEN = 2304
TESTS_DIR = Path(__file__).resolve().parent
VECTOR_DIR = TESTS_DIR / "test-vectors" / "mellum-llama-cpp"
ORACLES = {
    0: {
        "rows": 26,
        "reference": VECTOR_DIR / "l-out-0.f32",
        "sha256": "5a99d699c83a1a5417f9175f46e30c343026767372546001a112aef34ce49243",
        "max_abs_limit": 6.0e-3,
        "rms_limit": 1.25e-4,
    },
    27: {
        "rows": 1,
        "reference": VECTOR_DIR / "l-out-27-last.f32",
        "sha256": "050436ca257f8ebe36656f32785b9649ddf8b8ad75a08ec47f005ea588b72ebb",
    },
    "27-tokenwise": {
        "rows": 1,
        "reference": VECTOR_DIR / "l-out-27-tokenwise.f32",
        "sha256": "4ae7a46e409d6e3bf760ef8f7c671f32ff16d0798b8cd3dd25fe5fcbb3f086fe",
        "max_abs_limit": 1.6,
        "rms_limit": 5.9e-2,
    },
    "logits-tokenwise": {
        "values": 98304,
        "reference": VECTOR_DIR / "result-output-tokenwise.f32",
        "sha256": "4ec7f9c838058fc267ec0a2f117aa4db523950df1621f61d25afcf63e3be2b1c",
        "max_abs_limit": 2.3e-2,
        "rms_limit": 1.19e-2,
    },
}


def read_f32(path: Path, expected_bytes: int, values: int) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(f"{path} has {len(payload)} bytes; expected {expected_bytes}")
    return struct.unpack(f"<{values}f", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="F32 output from a Mellum diagnostic probe")
    parser.add_argument(
        "--layer",
        choices=("0", "27", "27-tokenwise", "logits-tokenwise"),
        default="0",
        help="checkpoint family (default: 0)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="override the pinned llama.cpp fixture",
    )
    args = parser.parse_args()
    layer = int(args.layer) if args.layer.isdecimal() else args.layer
    oracle = ORACLES[layer]
    reference_path = args.reference or oracle["reference"]
    values = oracle["values"] if "values" in oracle else oracle["rows"] * HIDDEN
    expected_bytes = values * 4

    try:
        reference_bytes = reference_path.read_bytes()
        reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
        if reference_sha256 != oracle["sha256"]:
            print(
                f"reference SHA-256 mismatch: {reference_sha256} != {oracle['sha256']}",
                file=sys.stderr,
            )
            return 1
        reference = read_f32(reference_path, expected_bytes, values)
        actual = read_f32(args.actual, expected_bytes, values)
    except (OSError, ValueError) as exc:
        print(f"oracle check failed: {exc}", file=sys.stderr)
        return 1

    max_abs = -1.0
    max_index = 0
    sum_abs = 0.0
    sum_sq = 0.0
    for index, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
        if not math.isfinite(expected):
            print(f"oracle check failed: non-finite reference value at index {index}", file=sys.stderr)
            return 1
        if not math.isfinite(observed):
            print(f"oracle check failed: non-finite actual value at index {index}", file=sys.stderr)
            return 1
        delta = observed - expected
        if not math.isfinite(delta):
            print(f"oracle check failed: non-finite delta at index {index}", file=sys.stderr)
            return 1
        absolute = abs(delta)
        if absolute > max_abs:
            max_abs = absolute
            max_index = index
        sum_abs += absolute
        sum_sq += delta * delta
    rms = math.sqrt(sum_sq / len(reference))
    mae = sum_abs / len(reference)
    if values % HIDDEN == 0:
        max_location = str(divmod(max_index, HIDDEN))
    else:
        max_location = str(max_index)
    print(
        f"Mellum layer-{args.layer} oracle "
        f"values={len(reference)} max_abs={max_abs:.9g} rms={rms:.9g} "
        f"mae={mae:.9g} max_at={max_location}"
    )
    max_abs_limit = oracle.get("max_abs_limit")
    rms_limit = oracle.get("rms_limit")
    if max_abs_limit is not None and (max_abs > max_abs_limit or rms > rms_limit):
        print(
            f"oracle gate failed: max_abs <= {max_abs_limit:g}, rms <= {rms_limit:g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
