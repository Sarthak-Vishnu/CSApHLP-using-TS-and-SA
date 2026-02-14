"""
Module: parameter_tuning_SA.py
Description: Parameter tuning for Simulated Annealing algorithm.

This module tests different parameter combinations for Simulated Annealing
and stores only one combined output CSV: SA_all_results.csv
"""

import os
import pandas as pd
from .algorithms import *
from .functions import *


def format_selected_hub_capacities(solution, w, dataset_name):
    """Format selected hub capacities as '4(L3); 3(L1); 7(L1)' for reporting."""
    hub_levels, infeasible = selected_hub_capacities(solution, w, dataset_name=dataset_name)
    if infeasible:
        return "INFEASIBLE"
    if not hub_levels:
        return "N/A"
    return "; ".join([f"{hub}({level})" for hub, level in hub_levels.items()])


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
    elif dataset_name == "CAB20":
        n = 20
        w_skiprows = 32
        c_skiprows = 56
        usecols = "B:U"
        nrows = 20
        sheet_name = "CAB 10, 20 and 25Nodes"
    elif dataset_name == "CAB25":
        n = 25
        w_skiprows = 80
        c_skiprows = 108
        usecols = "B:Z"
        nrows = 25
        sheet_name = "CAB 10, 20 and 25Nodes"
    elif dataset_name == "TR40":
        n = 40
        w_skiprows = 4
        c_skiprows = 47
        usecols = "C:AP"
        nrows = 40
        sheet_name = "TR 40 and 55 Nodes"
    elif dataset_name == "TR55":
        n = 55
        w_skiprows = 94
        c_skiprows = 152
        usecols = "C:BE"
        nrows = 55
        sheet_name = "TR 40 and 55 Nodes"
    elif dataset_name in ["RGP100", "RGP00"]:
        n = 100
        w_skiprows = 4
        c_skiprows = 108
        usecols = "B:CW"
        nrows = 100
        sheet_name = "RGP100"
    else:
        raise ValueError("Invalid dataset name. Choose CAB10/CAB20/CAB25/TR40/TR55/RGP100")

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


def save_results(results_df, filename="SA_all_results.csv"):
    """Save combined SA tuning results to a single CSV in results directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, "results")
    output_path = os.path.join(results_dir, filename)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def test_sa_parameters(dataset_name, p, sa_param_grid, num_runs=3):
    """Test SA parameter combinations for one dataset/p combination."""
    w, c, n = load_dataset(dataset_name)

    fixed_max_iterations = {
        "CAB10": 500,
        "CAB20": 500,
        "CAB25": 800,
        "TR40": 1000,
        "TR55": 1000,
        "RGP100": 1000,
    }

    alpha = 0.3
    max_iterations = fixed_max_iterations[dataset_name]

    results = []
    total_tests = len(sa_param_grid)
    test_count = 0

    print(f"\n{'='*70}")
    print(f"Testing {dataset_name} - p={p}, max_iterations={max_iterations}, alpha={alpha}")
    print(f"{'='*70}")

    for params in sa_param_grid:
        test_count += 1
        initial_temp = params["initial_temp"]
        delta = params["delta"]
        p0 = params["p0"]
        n_samples = params["n_samples"]

        print(f"\n[{test_count}/{total_tests}] Testing initial_temp={initial_temp}, delta={delta}, p0={p0}, n_samples={n_samples}...")

        costs = []
        times = []
        total_iterations_list = []
        solutions = []

        for _ in range(num_runs):
            solution, cost, exec_time, total_iterations = simulated_annealing(
                n=n,
                p=p,
                w=w,
                c=c,
                alpha=alpha,
                initial_temp=initial_temp,
                delta=delta,
                max_iterations=max_iterations,
                p0=p0,
                n_samples=n_samples,
                dataset_name=dataset_name,
            )
            costs.append(cost)
            times.append(exec_time)
            total_iterations_list.append(total_iterations)
            solutions.append(solution)

        best_cost = min(costs)
        best_cost_idx = costs.index(best_cost)
        best_solution = solutions[best_cost_idx]
        selected_capacities = format_selected_hub_capacities(best_solution, w, dataset_name)
        total_iterations_for_best_cost = total_iterations_list[best_cost_idx]

        avg_cost = sum(costs) / len(costs)
        std_cost = (sum((x - avg_cost) ** 2 for x in costs) / len(costs)) ** 0.5
        avg_time = sum(times) / len(times)
        avg_total_iterations = sum(total_iterations_list) / len(total_iterations_list)

        results.append(
            {
                "dataset": dataset_name,
                "p": p,
                "alpha": alpha,
                "max_iterations": max_iterations,
                "initial_temp": initial_temp,
                "delta": delta,
                "p0": p0,
                "n_samples": n_samples,
                "best_solution": best_solution,
                "selected_hub_capacities": selected_capacities,
                "best_cost": best_cost,
                "avg_cost": avg_cost,
                "std_cost": std_cost,
                "avg_time": avg_time,
                "avg_total_iterations": avg_total_iterations,
                "total_iterations_for_best_cost": total_iterations_for_best_cost,
            }
        )

        print(f"  Best Solution: {best_solution}")
        print(f"  Selected Hub Capacities: {selected_capacities}")
        print(f"  Best Cost: {best_cost:.4f}, Avg Cost: {avg_cost:.4f}, "
              f"Std Dev: {std_cost:.4f}, Avg Time: {avg_time:.4f}s, "
              f"Avg Total Iterations: {avg_total_iterations:.1f}, "
              f"# of Iterations (best cost): {total_iterations_for_best_cost}")

    return pd.DataFrame(results)


def build_sa_grid(dataset_name):
    """Create SA tuning grid where delta is the tuned parameter per dataset."""
    delta_candidates = {
        "CAB10": [0.01, 0.05, 0.1],
        "CAB20": [0.01, 0.05, 0.1],
        "CAB25": [0.01, 0.05, 0.1],
        "TR40": [0.01, 0.03, 0.05],
        "TR55": [0.005, 0.01, 0.05],
        "RGP100": [0.005, 0.01, 0.03],
    }

    # Keep other SA controls fixed while tuning delta
    if dataset_name in ["CAB10", "CAB20", "CAB25"]:
        p0 = 0.8
        n_samples = 50
    else:
        p0 = 0.8
        n_samples = 60

    return [
        {
            "initial_temp": None,
            "delta": delta,
            "p0": p0,
            "n_samples": n_samples,
        }
        for delta in delta_candidates[dataset_name]
    ]


def main():
    """Run SA tuning and save one combined CSV only."""
    all_results = []

    alpha = 0.3
    fixed_max_iterations = {
        "CAB10": 500,
        "CAB20": 500,
        "CAB25": 800,
        "TR40": 1000,
        "TR55": 1000,
        "RGP100": 1000,
    }

    dataset_p_pairs = [
        ("CAB10", 3),
        ("CAB10", 5),
        # ("CAB20", 3),
        # ("CAB20", 5),
        # ("CAB25", 3),
        # ("CAB25", 5),
        # ("TR40", 4),
        # ("TR40", 6),
        # ("TR55", 5),
        # ("TR55", 7),
        # ("RGP100", 9),
        # ("RGP100", 12),
    ]

    print("\n" + "="*70)
    print("SIMULATED ANNEALING PARAMETER TUNING")
    print("="*70)
    print(f"\nFixed Parameters:")
    print(f"  Alpha: {alpha}")
    print(f"  Max Iterations - CAB10: {fixed_max_iterations['CAB10']}, "
          f"CAB20: {fixed_max_iterations['CAB20']}, "
          f"CAB25: {fixed_max_iterations['CAB25']}, "
          f"TR40: {fixed_max_iterations['TR40']}, "
          f"TR55: {fixed_max_iterations['TR55']}, "
          f"RGP100: {fixed_max_iterations['RGP100']}")

    for dataset_name, p in dataset_p_pairs:
        sa_grid = build_sa_grid(dataset_name)
        result_df = test_sa_parameters(
            dataset_name=dataset_name,
            p=p,
            sa_param_grid=sa_grid,
            num_runs=3,
        )
        all_results.append(result_df)

    combined_results = pd.concat(all_results, ignore_index=True)
    save_results(combined_results, filename="SA_all_results.csv")

    print("\n" + "=" * 70)
    print("BEST SA PARAMETERS SUMMARY")
    print("=" * 70)

    summary_p_values = {
        "CAB10": [3, 5],
        "CAB20": [3, 5],
        "CAB25": [3, 5],
        "TR40": [4, 6],
        "TR55": [5, 7],
        "RGP100": [9, 12],
    }

    for dataset in ["CAB10", "CAB20", "CAB25", "TR40", "TR55", "RGP100"]:
        for p in summary_p_values[dataset]:
            subset = combined_results[(combined_results["dataset"] == dataset) &
                                     (combined_results["p"] == p)]
            if len(subset) > 0:
                best_idx = subset["avg_cost"].idxmin()
                best = combined_results.loc[best_idx]
                print(f"\n{dataset} - p={best['p']}")
                print(f"  Best SA Params: initial_temp={best['initial_temp']}, "
                      f"delta={best['delta']}, p0={best['p0']}, n_samples={int(best['n_samples'])}")
                print(f"  Best Solution: {best['best_solution']}")
                if "selected_hub_capacities" in best:
                    print(f"  Selected Hub Capacities: {best['selected_hub_capacities']}")
                print(f"  Best Cost: {best['best_cost']:.4f}")
                print(f"  Avg Cost: {best['avg_cost']:.4f}")
                print(f"  Std Dev: {best['std_cost']:.4f}")
                print(f"  Avg Time: {best['avg_time']:.4f}s")
                if "total_iterations_for_best_cost" in best:
                    print(f"  # of Iterations (best cost): {int(best['total_iterations_for_best_cost'])}")

    print("\n" + "=" * 70)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 70)
    best_overall_idx = combined_results["avg_cost"].idxmin()
    best_overall = combined_results.loc[best_overall_idx]
    print(f"Dataset: {best_overall['dataset']}")
    print(f"p: {int(best_overall['p'])}")
    print(f"max_iterations: {int(best_overall['max_iterations'])}")
    print(f"alpha: {best_overall['alpha']}")
    print(f"initial_temp: {best_overall['initial_temp']}")
    print(f"delta: {best_overall['delta']}")
    print(f"p0: {best_overall['p0']}")
    print(f"n_samples: {int(best_overall['n_samples'])}")
    print(f"Best Solution: {best_overall['best_solution']}")
    if "selected_hub_capacities" in best_overall:
        print(f"Selected Hub Capacities: {best_overall['selected_hub_capacities']}")
    print(f"Average Cost: {best_overall['avg_cost']:.4f}")
    print(f"Std Dev: {best_overall['std_cost']:.4f}")
    print(f"Avg Time: {best_overall['avg_time']:.4f}s")
    if "total_iterations_for_best_cost" in best_overall:
        print(f"# of Iterations (best cost): {int(best_overall['total_iterations_for_best_cost'])}")


if __name__ == "__main__":
    main()
