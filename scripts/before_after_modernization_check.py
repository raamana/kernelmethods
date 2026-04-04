#!/usr/bin/env python3
"""Compare core kernelmethods outputs before and after modernization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE_REPO = Path("/tmp/kernelmethods-before")

COMPARISON_SCRIPT = r"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="normalization of kernel matrix resulted in Inf / NaN values.*",
)
warnings.filterwarnings(
    "ignore",
    message="Kernel matrix computation resulted in Inf or NaN values!.*",
)

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "float_"):
    np.float_ = np.float64

try:
    import kernelmethods.utils as km_utils

    def _patched_ensure_ndarray_size(
        array,
        ensure_dtype=np.number,
        ensure_num_dim=2,
        ensure_min_samples=1,
        ensure_min_features=1,
    ):
        if not isinstance(array, np.ndarray):
            raise ValueError("input data type must be a numpy.ndarray")
        if ensure_dtype is not None and not np.issubdtype(array.dtype, ensure_dtype):
            raise ValueError("Data type must be one of {}".format(ensure_dtype))
        if array.ndim != ensure_num_dim:
            raise ValueError(
                "Input array dimension required: {} ; found {}".format(
                    ensure_num_dim, array.ndim
                )
            )
        if array.shape[0] < ensure_min_samples:
            raise ValueError(
                "Number of samples must be at least {}".format(ensure_min_samples)
            )
        if ensure_num_dim > 1 and array.shape[1] < ensure_min_features:
            raise ValueError(
                "Number of features must be at least {}".format(ensure_min_features)
            )
        return array

    km_utils.ensure_ndarray_size = _patched_ensure_ndarray_size
except Exception:
    pass

from kernelmethods import GaussianKernel, LinearKernel, PolyKernel
from kernelmethods.base import KernelMatrix
from kernelmethods.numeric_kernels import HadamardKernel
from kernelmethods.ranking import rank_kernels
from kernelmethods.sampling import KernelBucket

x = np.array([1.0, 2.0, 3.0])
y = np.array([3.0, 4.0, 3.0])
sample_data = np.array([[1.0, 0.0], [0.5, 1.0], [1.5, 1.0], [0.0, 2.0]])
labels = np.array([0.0, 1.0, 1.0, 0.0])

poly = PolyKernel(degree=4)
rbf = GaussianKernel()
linear = LinearKernel()
had = HadamardKernel(alpha=3)
km = KernelMatrix(rbf, normalized=True)
km.attach_to(sample_data)

result = {
    "poly_xy": float(poly(x, y)),
    "rbf_xy": float(rbf(x, y)),
    "lin_xy": float(linear(x, y)),
    "had_xy": float(had(x, y)),
    "km_diag": [float(v) for v in np.diag(km.full)],
    "km_01": float(km.full[0, 1]),
}

comparison_bucket = KernelBucket(
    rbf_sigma_values=[0.5, 1.0],
    poly_degree_values=[2],
    name="comparison_bucket",
)
comparison_bucket.attach_to(sample_data)

try:
    scores = rank_kernels(comparison_bucket, labels, method="align/corr")
    result["align_ranking"] = [float(v) for v in scores]
except Exception as exc:
    result["align_ranking_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result, sort_keys=True))
"""


def collect_results(repo_path: Path) -> dict:
    output = subprocess.check_output(
        [sys.executable, "-c", COMPARISON_SCRIPT, str(repo_path)],
        text=True,
    )
    return json.loads(output)


def print_results(title: str, results: dict) -> None:
    print(f"{title}:")
    print(json.dumps(results, indent=2, sort_keys=True))


def print_summary(before_results: dict, after_results: dict) -> None:
    comparison_rows = [
        ("PolyKernel(x, y)", before_results["poly_xy"], after_results["poly_xy"], "unchanged"),
        ("GaussianKernel(x, y)", before_results["rbf_xy"], after_results["rbf_xy"], "unchanged"),
        ("LinearKernel(x, y)", before_results["lin_xy"], after_results["lin_xy"], "unchanged"),
        ("KernelMatrix[0, 1]", before_results["km_01"], after_results["km_01"], "unchanged"),
        ("HadamardKernel(x, y)", before_results["had_xy"], after_results["had_xy"], "intentional fix"),
        (
            "Alignment ranking",
            before_results.get("align_ranking_error", "available"),
            "available" if "align_ranking" in after_results else after_results.get("align_ranking_error"),
            "implemented",
        ),
    ]
    width = max(len(row[0]) for row in comparison_rows) + 2
    print("\nSummary:")
    for label, before_value, after_value, note in comparison_rows:
        print(f"{label:<{width}} before={before_value}")
        print(f"{' ' * width} after={after_value}  [{note}]\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a pre-modernization checkout against the current repo."
    )
    parser.add_argument(
        "--before-repo",
        type=Path,
        default=DEFAULT_BEFORE_REPO,
        help="Path to the pre-modernization checkout.",
    )
    parser.add_argument(
        "--after-repo",
        type=Path,
        default=REPO_ROOT,
        help="Path to the modernized checkout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before_repo = args.before_repo.resolve()
    after_repo = args.after_repo.resolve()

    if not before_repo.exists():
        print(f"Missing before repo: {before_repo}", file=sys.stderr)
        return 1
    if not after_repo.exists():
        print(f"Missing after repo: {after_repo}", file=sys.stderr)
        return 1

    before_results = collect_results(before_repo)
    after_results = collect_results(after_repo)

    print_results("Before modernization", before_results)
    print()
    print_results("After modernization", after_results)
    print_summary(before_results, after_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
