"""
Estimate how many random seeds are needed to detect a given accuracy difference
between GE-CNN and CNN from an existing param_datasize_results.pt file.

Default use:
    python required_seeds_only.py

Output:
    Required seeds: X

The script assumes the result file has the structure created by
`test_parameters_datasize_v2.py`, i.e. entries with `ge_accs` and `cnn_accs`.
By default it uses Experiment A only, because that is the parameter sweep.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def z_value_for_common_settings(alpha: float, power: float, two_sided: bool = True):
    """
    Return z_alpha and z_power.

    For the default alpha=0.05, power=0.80, two-sided test:
        z_alpha ≈ 1.96
        z_power ≈ 0.84

    Uses scipy if available. Otherwise falls back to common defaults.
    """
    try:
        from scipy import stats
        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_power = stats.norm.ppf(power)
        return float(z_alpha), float(z_power)
    except Exception:
        if alpha == 0.05 and power == 0.80 and two_sided:
            return 1.959963984540054, 0.8416212335729143
        if alpha == 0.05 and power == 0.80 and not two_sided:
            return 1.6448536269514722, 0.8416212335729143
        raise RuntimeError(
            "scipy is required for non-default alpha/power settings. "
            "Install scipy or use alpha=0.05 and power=0.80."
        )


def required_seeds_from_pair(ge_accs, cnn_accs, delta: float, alpha: float, power: float, two_sided: bool):
    """
    Estimate required seeds for a paired comparison.

    We use seed-wise differences:
        d_i = acc_GE_i - acc_CNN_i

    Then estimate:
        n ≈ ((z_alpha + z_power) * sd(d_i) / delta)^2
    """
    ge = np.asarray(ge_accs, dtype=float)
    cnn = np.asarray(cnn_accs, dtype=float)

    if len(ge) != len(cnn):
        raise ValueError("ge_accs and cnn_accs must have the same length.")
    if len(ge) < 2:
        raise ValueError("At least 2 pilot seeds are needed to estimate seed variation.")

    diffs = ge - cnn
    sd_diff = float(np.std(diffs, ddof=1))

    if sd_diff == 0:
        return 2

    z_alpha, z_power = z_value_for_common_settings(alpha, power, two_sided)
    n = ((z_alpha + z_power) * sd_diff / delta) ** 2

    return max(2, int(math.ceil(n)))


def collect_rows(results, experiment: str):
    if experiment == "exp_a":
        return results.get("exp_a", [])
    if experiment == "exp_b":
        return results.get("exp_b", [])
    if experiment == "all":
        return results.get("exp_a", []) + results.get("exp_b", [])
    raise ValueError("experiment must be one of: exp_a, exp_b, all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/param_datasize_results.pt",
        help="Path to the .pt result file from the parameter/data-size experiment.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="Smallest accuracy difference you want to detect. Default: 0.02.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level. Default: 0.05.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired statistical power. Default: 0.80.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["exp_a", "exp_b", "all"],
        default="exp_a",
        help="Which experiment to use. Default: exp_a, the parameter sweep.",
    )
    parser.add_argument(
        "--one-sided",
        action="store_true",
        help="Use a one-sided test. Default is two-sided.",
    )
    args = parser.parse_args()

    path = Path(args.results_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find result file: {path}")

    results = torch.load(path, map_location="cpu")
    rows = collect_rows(results, args.experiment)

    if not rows:
        raise ValueError(f"No rows found for experiment={args.experiment}")

    required = []
    for row in rows:
        if "ge_accs" not in row or "cnn_accs" not in row:
            continue
        n = required_seeds_from_pair(
            row["ge_accs"],
            row["cnn_accs"],
            delta=args.delta,
            alpha=args.alpha,
            power=args.power,
            two_sided=not args.one_sided,
        )
        required.append(n)

    if not required:
        raise ValueError("No rows with ge_accs and cnn_accs were found.")

    # Conservative choice: enough seeds for the noisiest parameter point.
    print(f"Required seeds: {max(required)}")


if __name__ == "__main__":
    main()
