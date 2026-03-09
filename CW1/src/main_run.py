"""
Module: main.py
Description: Main execution file for CSApHLP using Tabu Search and Simulated Annealing.
             Runs each algorithm 5 times per instance and reports results per Table 1
             of the assignment brief.
"""

import pandas as pd
import os
import time

try:
    from .algorithms import tabu_search, simulated_annealing
    from .functions import selected_hub_capacities, normalize_flow
except ImportError:
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.algorithms import tabu_search, simulated_annealing
    from src.functions import selected_hub_capacities, normalize_flow


# ---------------------------------------------------------------------------
# Tuned parameters (selected from parameter tuning experiments)
# ---------------------------------------------------------------------------

TABU_PARAMS = {
    ('CAB10',   3): {'tabu_tenure':  5, 'max_iterations':   200},
    ('CAB10',   4): {'tabu_tenure':  9, 'max_iterations':   200},
    ('CAB20',   3): {'tabu_tenure':  6, 'max_iterations':   500},
    ('CAB20',   5): {'tabu_tenure':  8, 'max_iterations':   500},
    ('CAB25',   3): {'tabu_tenure':  8, 'max_iterations':   800},
    ('CAB25',   5): {'tabu_tenure':  8, 'max_iterations':   800},
    ('TR40',    4): {'tabu_tenure':  8, 'max_iterations':  1000},
    ('TR40',    6): {'tabu_tenure': 12, 'max_iterations':  1000},
    ('TR55',    5): {'tabu_tenure': 13, 'max_iterations':  1000},
    ('TR55',    7): {'tabu_tenure': 15, 'max_iterations':  1000},
    ('RGP100',  9): {'tabu_tenure': 17, 'max_iterations':  1000},
    ('RGP100', 12): {'tabu_tenure': 24, 'max_iterations':  1000},
}

SA_PARAMS = {
    ('CAB10',   3): {'beta': 0.990, 'max_iterations':  1000},
    ('CAB10',   4): {'beta': 0.999, 'max_iterations':  1000},
    ('CAB20',   3): {'beta': 0.995, 'max_iterations':  2000},
    ('CAB20',   5): {'beta': 0.970, 'max_iterations':  2000},
    ('CAB25',   3): {'beta': 0.850, 'max_iterations':  2500},
    ('CAB25',   5): {'beta': 0.960, 'max_iterations':  2500},
    ('TR40',    4): {'beta': 0.990, 'max_iterations':  4000},
    ('TR40',    6): {'beta': 0.980, 'max_iterations':  4000},
    ('TR55',    5): {'beta': 0.995, 'max_iterations':  5500},
    ('TR55',    7): {'beta': 0.995, 'max_iterations':  5500},
    ('RGP100',  9): {'beta': 0.995, 'max_iterations': 10000},
    ('RGP100', 12): {'beta': 0.990, 'max_iterations': 10000},
}

# Assignment brief instances: (dataset, p, alpha)
# RGP100 uses alpha=0.3 only; all others use alpha=0.3 and alpha=0.7
INSTANCES = [
    ('CAB10',  3, 0.3), ('CAB10',  3, 0.7),
    ('CAB10',  4, 0.3), ('CAB10',  4, 0.7),
    ('CAB20',  3, 0.3), ('CAB20',  3, 0.7),
    ('CAB20',  5, 0.3), ('CAB20',  5, 0.7),
    ('CAB25',  3, 0.3), ('CAB25',  3, 0.7),
    ('CAB25',  5, 0.3), ('CAB25',  5, 0.7),
    ('TR40',   4, 0.3), ('TR40',   4, 0.7),
    ('TR40',   6, 0.3), ('TR40',   6, 0.7),
    ('TR55',   5, 0.3), ('TR55',   5, 0.7),
    ('TR55',   7, 0.3), ('TR55',   7, 0.7),
    ('RGP100', 9, 0.3),
    ('RGP100',12, 0.3),
]

# Large instances: report hub locations only (no full allocation vector)
LARGE_DATASETS = {'TR40', 'TR55', 'RGP100'}

NUM_RUNS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset_name):
    """Load flow matrix w and cost matrix c for a given dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'CAB_TR_and RGP Datasets_2026.xlsx')

    configs = {
        'CAB10':  dict(n=10,  sheet="CAB 10, 20 and 25Nodes", w_skip=2,   c_skip=18,  cols="B:K",  nrows=10),
        'CAB20':  dict(n=20,  sheet="CAB 10, 20 and 25Nodes", w_skip=32,  c_skip=56,  cols="B:U",  nrows=20),
        'CAB25':  dict(n=25,  sheet="CAB 10, 20 and 25Nodes", w_skip=80,  c_skip=108, cols="B:Z",  nrows=25),
        'TR40':   dict(n=40,  sheet="TR 40 and 55 Nodes",     w_skip=4,   c_skip=47,  cols="C:AP", nrows=40),
        'TR55':   dict(n=55,  sheet="TR 40 and 55 Nodes",     w_skip=94,  c_skip=152, cols="C:BE", nrows=55),
        'RGP100': dict(n=100, sheet="RGP100",                 w_skip=4,   c_skip=108, cols="B:CW", nrows=100),
    }
    cfg = configs[dataset_name]

    w = pd.read_excel(file_path, sheet_name=cfg['sheet'], header=None,
                      usecols=cfg['cols'], skiprows=cfg['w_skip'], nrows=cfg['nrows'])
    w = normalize_flow(w)

    c = pd.read_excel(file_path, sheet_name=cfg['sheet'], header=None,
                      usecols=cfg['cols'], skiprows=cfg['c_skip'], nrows=cfg['nrows'])
    c = c.apply(pd.to_numeric, errors='coerce')

    return w, c, cfg['n']


def format_solution(solution, dataset_name):
    """
    For small instances: return full allocation vector.
    For large instances: return sorted hub node IDs only.
    """
    if dataset_name in LARGE_DATASETS:
        hubs = sorted(set(solution))
        return str(hubs)
    return str(solution)


def format_hub_capacities(solution, w, dataset_name):
    """Return hub capacity string e.g. '4(L3); 3(L1); 7(L1)', or INFEASIBLE."""
    hub_levels, infeasible = selected_hub_capacities(solution, w, dataset_name=dataset_name)
    if infeasible:
        return "INFEASIBLE"
    if not hub_levels:
        return "N/A"
    return "; ".join(f"{h}({lv})" for h, lv in hub_levels.items())


def run_algorithm(algo, dataset_name, p, alpha, w, c, n):
    """
    Run a given algorithm NUM_RUNS times and return aggregated results.

    Args:
        algo (str): 'TS' or 'SA'
        dataset_name, p, alpha, w, c, n: instance data

    Returns:
        dict of aggregated results
    """
    costs, times, iterations, solutions = [], [], [], []

    for run in range(1, NUM_RUNS + 1):
        print(f"    Run {run}/{NUM_RUNS}...", end=" ", flush=True)

        if algo == 'TS':
            params = TABU_PARAMS[(dataset_name, p)]
            solution, cost, exec_time, total_iter = tabu_search(
                n=n, p=p, w=w, c=c, alpha=alpha,
                max_iterations=params['max_iterations'],
                tabu_tenure=params['tabu_tenure'],
                dataset_name=dataset_name,
            )
        else:  # SA
            params = SA_PARAMS[(dataset_name, p)]
            solution, cost, exec_time, total_iter = simulated_annealing(
                n=n, p=p, w=w, c=c, alpha=alpha,
                beta=params['beta'],
                max_iterations=params['max_iterations'],
                p0=0.8,
                n_samples=50,
                dataset_name=dataset_name,
            )

        costs.append(cost)
        times.append(exec_time)
        iterations.append(total_iter)
        solutions.append(solution)
        print(f"cost={cost:.4f}")

    best_idx   = costs.index(min(costs))
    best_cost  = costs[best_idx]
    best_sol   = solutions[best_idx]
    avg_cost   = sum(costs) / NUM_RUNS
    std_cost   = (sum((x - avg_cost) ** 2 for x in costs) / NUM_RUNS) ** 0.5
    avg_time   = sum(times) / NUM_RUNS
    iter_best  = iterations[best_idx]

    return {
        'algorithm':              algo,
        'dataset':                dataset_name,
        'p':                      p,
        'alpha':                  alpha,
        'solution_config':        format_solution(best_sol, dataset_name),
        'selected_hub_capacities': format_hub_capacities(best_sol, w, dataset_name),
        'best_cost':              best_cost,
        'avg_cost':               avg_cost,
        'std_cost':               std_cost,
        'avg_time':               avg_time,
        'iter_at_best':           iter_best,
    }


def save_results(results_df, filename):
    """Save results CSV to the results directory."""
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    out_path     = os.path.join(project_root, 'results', filename)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


def print_summary_table(results_df, algo_label):
    """Print a formatted summary table matching assignment brief Table 1 layout."""
    col_w = {
        'problem': 10, 'p': 4, 'alpha': 6,
        'config':  45, 'tnc': 18, 'avg': 18, 'time': 16,
    }

    header = (
        f"{'Problem':<{col_w['problem']}} {'p':<{col_w['p']}} {'α':<{col_w['alpha']}} "
        f"{'Solution Config / Hub Locations':<{col_w['config']}} "
        f"{'TNC':>{col_w['tnc']}} {'Avg TNC':>{col_w['avg']}} "
        f"{'Time (s) / Iter #':>{col_w['time']}}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"COMPUTATIONAL RESULTS — {algo_label}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for _, row in results_df.iterrows():
        config_str = row['solution_config']
        # Truncate long config strings for display
        if len(config_str) > col_w['config']:
            config_str = config_str[:col_w['config'] - 3] + "..."
        time_iter = f"{row['avg_time']:.3f}s / {int(row['iter_at_best'])}"
        print(
            f"{row['dataset']:<{col_w['problem']}} "
            f"{int(row['p']):<{col_w['p']}} "
            f"{row['alpha']:<{col_w['alpha']}} "
            f"{config_str:<{col_w['config']}} "
            f"{row['best_cost']:>{col_w['tnc']}.4f} "
            f"{row['avg_cost']:>{col_w['avg']}.4f} "
            f"{time_iter:>{col_w['time']}}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_ts_results = []
    all_sa_results = []

    print("\n" + "=" * 70)
    print("CSApHLP — MAIN COMPUTATIONAL EXPERIMENT")
    print(f"Algorithms: Tabu Search (TS) | Simulated Annealing (SA)")
    print(f"Runs per instance: {NUM_RUNS}")
    print(f"Total instances: {len(INSTANCES)}")
    print("=" * 70)

    # Cache loaded datasets to avoid repeated I/O
    dataset_cache = {}

    for dataset_name, p, alpha in INSTANCES:
        print(f"\n[{dataset_name} | p={p} | α={alpha}]")

        # Load dataset once per dataset name
        if dataset_name not in dataset_cache:
            print(f"  Loading {dataset_name}...")
            dataset_cache[dataset_name] = load_dataset(dataset_name)
        w, c, n = dataset_cache[dataset_name]

        # --- Tabu Search ---
        print(f"  Tabu Search (tenure={TABU_PARAMS[(dataset_name, p)]['tabu_tenure']}, "
              f"max_iter={TABU_PARAMS[(dataset_name, p)]['max_iterations']}):")
        ts_result = run_algorithm('TS', dataset_name, p, alpha, w, c, n)
        all_ts_results.append(ts_result)
        print(f"    Best Cost : {ts_result['best_cost']:.4f}")
        print(f"    Avg  Cost : {ts_result['avg_cost']:.4f}  (std={ts_result['std_cost']:.4f})")
        print(f"    Hub Config: {ts_result['selected_hub_capacities']}")

        # --- Simulated Annealing ---
        print(f"  Simulated Annealing (beta={SA_PARAMS[(dataset_name, p)]['beta']}, "
              f"max_iter={SA_PARAMS[(dataset_name, p)]['max_iterations']}):")
        sa_result = run_algorithm('SA', dataset_name, p, alpha, w, c, n)
        all_sa_results.append(sa_result)
        print(f"    Best Cost : {sa_result['best_cost']:.4f}")
        print(f"    Avg  Cost : {sa_result['avg_cost']:.4f}  (std={sa_result['std_cost']:.4f})")
        print(f"    Hub Config: {sa_result['selected_hub_capacities']}")

    # --- Combine and save ---
    ts_df = pd.DataFrame(all_ts_results)
    sa_df = pd.DataFrame(all_sa_results)
    combined_df = pd.concat([ts_df, sa_df], ignore_index=True)

    save_results(ts_df,       'main_TS_results.csv')
    save_results(sa_df,       'main_SA_results.csv')
    save_results(combined_df, 'main_all_results.csv')

    # --- Print summary tables ---
    print_summary_table(ts_df, "Tabu Search")
    print_summary_table(sa_df, "Simulated Annealing")

    # --- Head-to-head comparison ---
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON (TS vs SA) — Best Cost")
    print(f"{'='*70}")
    print(f"{'Instance':<20} {'TS Best Cost':>18} {'SA Best Cost':>18} {'Winner':>10}")
    print("-" * 70)
    for ts_row, sa_row in zip(all_ts_results, all_sa_results):
        label   = f"{ts_row['dataset']} p={ts_row['p']} α={ts_row['alpha']}"
        ts_cost = ts_row['best_cost']
        sa_cost = sa_row['best_cost']
        if ts_cost < sa_cost:
            winner = "TS"
        elif sa_cost < ts_cost:
            winner = "SA"
        else:
            winner = "Tie"
        print(f"{label:<20} {ts_cost:>18.4f} {sa_cost:>18.4f} {winner:>10}")
    print("-" * 70)


if __name__ == "__main__":
    main()