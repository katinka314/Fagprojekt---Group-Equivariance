#!/usr/bin/env python3
"""
Compute required number of seeds for GE-CNN vs CNN experiments.

The script computes both:
  1. Worst-case required seeds: max over all settings.
  2. Average-case required seeds: average over settings.

It works with result files that contain entries with:
  - ge_accs
  - cnn_accs

Example:
  python seed_power_worst_average.py
  python seed_power_worst_average.py --results-path results/param_datasize_results.pt

Default assumptions:
  - delta = 0.02 accuracy difference
  - alpha = 0.05
  - power = 0.80
  - two-sided test
"""

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch


def z_values(alpha: float, power: float, alternative: str):
    """Return z_alpha and z_power for approximate normal power calculation."""
    nd = NormalDist()

    if alternative == "two-sided":
        z_alpha = nd.inv_cdf(1 - alpha / 2)
    elif alternative in {"greater", "less", "one-sided"}:
        z_alpha = nd.inv_cdf(1 - alpha)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', 'less', or 'one-sided'")

    z_power = nd.inv_cdf(power)
    return z_alpha, z_power


def required_seeds_from_sd(sd_diff: float, delta: float, alpha: float, power: float, alternative: str):
    """Approximate required seeds to detect delta given sd of paired differences."""
    if sd_diff <= 0:
        return 1

    z_alpha, z_power = z_values(alpha, power, alternative)
    n = ((z_alpha + z_power) * sd_diff / delta) ** 2
    return int(math.ceil(n))


def analyze_records(records, delta: float, alpha: float, power: float, alternative: str):
    """Analyze a list of experiment records with ge_accs and cnn_accs."""
    rows = []

    for idx, r in enumerate(records):
        if "ge_accs" not in r or "cnn_accs" not in r:
            continue

        ge = np.asarray(r["ge_accs"], dtype=float)
        cnn = np.asarray(r["cnn_accs"], dtype=float)

        if len(ge) != len(cnn):
            raise ValueError(f"Record {idx}: ge_accs and cnn_accs have different lengths")

        if len(ge) < 2:
            raise ValueError(f"Record {idx}: need at least 2 seeds to estimate standard deviation")

        diffs = ge - cnn
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        n_required = required_seeds_from_sd(sd_diff, delta, alpha, power, alternative)

        if "cnn_channels" in r:
            setting = f"cnn_channels={r['cnn_channels']}"
        elif "fraction" in r:
            setting = f"fraction={r['fraction']}"
        else:
            setting = f"setting_{idx}"

        rows.append({
            "setting": setting,
            "n_current": len(diffs),
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "required_seeds": n_required,
            "raw_record": r,
        })

    if not rows:
        return None

    worst = max(rows, key=lambda x: x["required_seeds"])
    average_required = int(math.ceil(np.mean([x["required_seeds"] for x in rows])))
    median_required = int(math.ceil(np.median([x["required_seeds"] for x in rows])))
    average_sd = float(np.mean([x["sd_diff"] for x in rows]))
    average_sd_required = required_seeds_from_sd(average_sd, delta, alpha, power, alternative)

    return {
        "rows": rows,
        "worst_case_required": worst["required_seeds"],
        "worst_case_setting": worst["setting"],
        "worst_case_sd": worst["sd_diff"],
        "average_case_required_mean_of_n": average_required,
        "median_case_required": median_required,
        "average_sd": average_sd,
        "average_case_required_from_average_sd": average_sd_required,
    }


def print_analysis(name: str, analysis):
    if analysis is None:
        print(f"\n{name}: no seed-wise ge_accs/cnn_accs found")
        return

    print(f"\n{name}")
    print("-" * len(name))
    print("Per setting:")
    for row in analysis["rows"]:
        print(
            f"  {row['setting']:>18} | "
            f"mean_diff={row['mean_diff']:+.4f} | "
            f"sd_diff={row['sd_diff']:.4f} | "
            f"required_seeds={row['required_seeds']}"
        )

    print("\nSummary:")
    print(f"  Worst-case required seeds: {analysis['worst_case_required']}")
    print(f"    driven by: {analysis['worst_case_setting']}")
    print(f"    sd_diff:   {analysis['worst_case_sd']:.4f}")
    print(f"  Average-case required seeds, mean of per-setting seed counts: {analysis['average_case_required_mean_of_n']}")
    print(f"  Median-case required seeds: {analysis['median_case_required']}")
    print(f"  Average-case required seeds, using average sd_diff: {analysis['average_case_required_from_average_sd']}")
    print(f"    average sd_diff: {analysis['average_sd']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, default="results/param_datasize_results.pt",
                        help="Path to .pt result file containing exp_a and/or exp_b")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="Minimum accuracy difference to detect, e.g. 0.02")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--power", type=float, default=0.80,
                        help="Desired statistical power")
    parser.add_argument("--alternative", type=str, default="two-sided",
                        choices=["two-sided", "greater", "less", "one-sided"],
                        help="Use two-sided for 'any difference', one-sided/greater for 'GE-CNN better than CNN'")
    args = parser.parse_args()

    path = Path(args.results_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find result file: {path}")

    data = torch.load(path, map_location="cpu")

    print("Seed power analysis")
    print("===================")
    print(f"File:        {path}")
    print(f"delta:       {args.delta}")
    print(f"alpha:       {args.alpha}")
    print(f"power:       {args.power}")
    print(f"alternative: {args.alternative}")

    exp_a = data.get("exp_a", [])
    exp_b = data.get("exp_b", [])

    analysis_a = analyze_records(exp_a, args.delta, args.alpha, args.power, args.alternative)
    analysis_b = analyze_records(exp_b, args.delta, args.alpha, args.power, args.alternative)

    print_analysis("Parameter experiment (exp_a)", analysis_a)
    print_analysis("Data-efficiency experiment (exp_b)", analysis_b)

    print("\nCompact result")
    print("--------------")
    if analysis_a is not None:
        print(f"Parameter worst-case: {analysis_a['worst_case_required']} seeds")
        print(f"Parameter average-case: {analysis_a['average_case_required_mean_of_n']} seeds")
    if analysis_b is not None:
        print(f"Data-efficiency worst-case: {analysis_b['worst_case_required']} seeds")
        print(f"Data-efficiency average-case: {analysis_b['average_case_required_mean_of_n']} seeds")


if __name__ == "__main__":
    main()
