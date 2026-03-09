"""
Module: debug_sa.py
Description: Debug script to plot Simulated Annealing cooling schedules.

This script reproduces the SA setup used in parameter tuning for CAB10 and
plots temperature cooling curves for p=3 and p=4 using the current
round-2/final beta candidates (8 curves total).
No files are written.
"""

import os
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})

try:
    from .algorithms import _compute_initial_temperature
    from .functions import normalize_flow, initial_solution_closest
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.algorithms import _compute_initial_temperature
    from src.functions import normalize_flow, initial_solution_closest


def load_dataset(dataset_name="CAB10"):
    """Load one dataset and return (w, c, n)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    file_path = os.path.join(data_dir, "CAB_TR_and RGP Datasets_2026.xlsx")

    if dataset_name == "CAB10":
        n = 10
        w_skiprows = 2
        c_skiprows = 18
        usecols = "B:K"
        nrows = 10
        sheet_name = "CAB 10, 20 and 25Nodes"
    else:
        raise ValueError("This debug script is configured for CAB10 only.")

    w = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        skiprows=w_skiprows,
        nrows=nrows,
    )
    w = normalize_flow(w)

    c = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        skiprows=c_skiprows,
        nrows=nrows,
    )
    c = c.apply(pd.to_numeric, errors="coerce")

    return w, c, n


def build_cab10_sa_grid(p):
    """Create CAB10 SA tuning grid aligned with parameter_tuning_SA.py."""
    beta_candidates_by_p = {
        3: [0.97, 0.98, 0.990, 0.995],
        4: [0.98, 0.99, 0.995, 0.999],
    }

    beta_candidates = beta_candidates_by_p.get(p, [])
    p0 = 0.8
    n_samples = 50

    return [
        {
            "initial_temp": None,
            "beta": beta,
            "p0": p0,
            "n_samples": n_samples,
        }
        for beta in beta_candidates
    ]


def build_temperature_curve(initial_temperature, beta, stage_length, max_iterations):
    """Build per-iteration geometric cooling curve (current SA implementation).

    This cools the temperature at every move (iteration) by multiplying
    by `beta` for exactly `max_iterations` iterations, matching
    algorithms.simulated_annealing.
    """
    temperature = max(float(initial_temperature), 1e-12)
    final_temperature = max(temperature * (beta ** max_iterations), 1e-300)

    iterations = [0]
    temperatures = [temperature]

    # Classical SA: update temperature every iteration (move)
    for iteration in range(1, int(max_iterations) + 1):
        temperature *= beta
        iterations.append(iteration)
        temperatures.append(temperature)

    return iterations, temperatures, final_temperature


def debug_sa_cooling():
    """Print SA setup and plot cooling curves for CAB10 with p=3 and p=4."""
    dataset_name = "CAB10"
    alpha = 0.3
    levels = 10
    sl = 10
    p_values = [3, 4]

    w, c, n = load_dataset(dataset_name)
    stage_length = sl * n
    max_iterations = levels * stage_length

    print("\n" + "=" * 70)
    print("SIMULATED ANNEALING COOLING DEBUG")
    print("=" * 70)
    print("\nFixed Parameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Alpha: {alpha}")
    print(f"  LEVELS: {levels}")
    print(f"  sl: {sl}")
    print(f"  stage_length = sl * n = {stage_length}")
    print(f"  max_iterations = LEVELS * stage_length = {max_iterations}")
    print("  p0: 0.8")
    print("  n_samples: 50")
    print("  Output: plot only (no files saved)")

    curves = []

    for p in p_values:
        sa_grid = build_cab10_sa_grid(p)
        print(f"\n{'='*70}")
        print(f"Testing {dataset_name} - p={p}, max_iterations={max_iterations}, alpha={alpha}")
        print(f"Using LEVELS={levels}, sl={sl}, stage_length={stage_length}")
        print(f"{'='*70}")

        total_tests = len(sa_grid)
        for idx, params in enumerate(sa_grid, start=1):
            beta = params["beta"]
            p0 = params["p0"]
            n_samples = params["n_samples"]

            print(
                f"\n[{idx}/{total_tests}] Testing initial_temp=None, beta={beta}, "
                f"sl={sl}, p0={p0}, n_samples={n_samples}..."
            )

            random.seed(1000 + p * 10 + idx)
            current_solution = initial_solution_closest(n, p, c)
            initial_temperature = _compute_initial_temperature(
                n=n,
                current_solution=current_solution,
                w=w,
                c=c,
                alpha=alpha,
                p0=p0,
                n_samples=n_samples,
                dataset_name=dataset_name,
            )

            iterations, temperatures, final_temperature = build_temperature_curve(
                initial_temperature=initial_temperature,
                beta=beta,
                stage_length=stage_length,
                max_iterations=max_iterations,
            )

            curves.append(
                {
                    "label": f"p={p}, beta={beta}",
                    "iterations": iterations,
                    "temperatures": temperatures,
                }
            )

            print(f"  Computed T0: {initial_temperature:.6f}")
            print(f"  Target Tf (T0 * beta^max_iterations): {final_temperature:.6f}")
            print(f"  Cooling iterations generated: {len(temperatures) - 1}")
            print(f"  Final plotted T: {temperatures[-1]:.6f}")

    plt.figure(figsize=(11, 7))
    for curve in curves:
        plt.plot(curve["iterations"], curve["temperatures"], label=curve["label"])

    # plt.title("SA Cooling Curves (CAB10): p=3 and p=4")
    plt.xlabel("Iterations", fontsize=22)
    plt.ylabel("Temperature", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    debug_sa_cooling()


if __name__ == "__main__":
    main()