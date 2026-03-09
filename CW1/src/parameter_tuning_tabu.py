"""
Module: parameter_tuning_tabu.py
Description: Parameter tuning for Tabu Search algorithm to find optimal settings

This module tests different parameter combinations for the Tabu Search algorithm
and identifies the best parameters using best_cost as the primary criterion,
with avg_cost used as a tie-breaker.
"""

import pandas as pd
import os
import time
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

def load_dataset(dataset_name='CAB10'):
    """
    Load dataset for parameter tuning
    
    Args:
        dataset_name (str): Name of dataset ('CAB10', 'CAB20', or 'CAB25')
        
    Returns:
        tuple: (w, c, n) where w is flow matrix, c is cost matrix, n is number of nodes
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    file_path = os.path.join(data_dir, 'CAB_TR_and RGP Datasets_2026.xlsx')
    
    if dataset_name == 'CAB10':
        n = 10
        w_skiprows = 2
        c_skiprows = 18
        usecols = "B:K"
        nrows = 10
        sheet_name = "CAB 10, 20 and 25Nodes"
    elif dataset_name == 'CAB20':
        n = 20
        w_skiprows = 32
        c_skiprows = 56
        usecols = "B:U"
        nrows = 20
        sheet_name = "CAB 10, 20 and 25Nodes"
    elif dataset_name == 'CAB25':
        n = 25
        w_skiprows = 80
        c_skiprows = 108
        usecols = "B:Z"
        nrows = 25
        sheet_name = "CAB 10, 20 and 25Nodes"
    elif dataset_name == 'TR40':
        n = 40
        w_skiprows = 4
        c_skiprows = 47
        usecols = "C:AP"
        nrows = 40
        sheet_name = "TR 40 and 55 Nodes"
    elif dataset_name == 'TR55':
        n = 55
        w_skiprows = 94
        c_skiprows = 152
        usecols = "C:BE"
        nrows = 55
        sheet_name = "TR 40 and 55 Nodes"
    elif dataset_name in ['RGP100', 'RGP00']:
        n = 100
        w_skiprows = 4
        c_skiprows = 108
        usecols = "B:CW"
        nrows = 100
        sheet_name = "RGP100"
    else:
        raise ValueError("Invalid dataset name. Choose 'CAB10', 'CAB20', 'CAB25', 'TR40', 'TR55', or 'RGP100'")
    
    # Load flow matrix
    w = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        skiprows=w_skiprows,
        nrows=nrows
    )
    w = normalize_flow(w)
    
    # Load cost matrix
    c = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        skiprows=c_skiprows,
        nrows=nrows
    )
    
    # Convert cost matrix to numeric values (in case they are strings)
    c = c.apply(pd.to_numeric, errors='coerce')
    
    return w, c, n


def save_results(results_df, filename):
    """
    Save results to CSV file in results directory
    
    Args:
        results_df (DataFrame): Results dataframe
        filename (str): Output filename
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up from src to CW1
    results_dir = os.path.join(project_root, 'results')
    output_path = os.path.join(results_dir, filename)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def test_fixed_parameters(dataset_name, p, tabu_tenure_range, num_runs=5):
    """
    Test tabu_tenure with fixed max_iterations and alpha for a specific dataset
    
    Args:
        dataset_name (str): 'CAB10', 'CAB20', or 'CAB25'
        p (int): Number of hubs
        tabu_tenure_range (list): List of tabu_tenure values to test
        num_runs (int): Number of runs for each configuration
        
    Returns:
        DataFrame: Results with tabu_tenure and performance metrics
    """
    # Load dataset
    w, c, n = load_dataset(dataset_name)
    
    # Fixed parameters per dataset
    fixed_params = {
        'CAB10': {'max_iterations': 200},
        'CAB20': {'max_iterations': 500},
        'CAB25': {'max_iterations': 800},
        'TR40': {'max_iterations': 1000},
        'TR55': {'max_iterations': 1000},
        'RGP100': {'max_iterations': 1000}
    }
    
    alpha = 0.3
    max_iterations = fixed_params[dataset_name]['max_iterations']
    
    results = []
    total_tests = len(tabu_tenure_range)
    test_count = 0
    
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name} - p={p}, max_iterations={max_iterations}, alpha={alpha}")
    print(f"{'='*70}")
    
    for tenure in tabu_tenure_range:
        test_count += 1
        print(f"\n[{test_count}/{total_tests}] Testing tabu_tenure={tenure}...")
        
        costs = []
        times = []
        total_iterations_list = []
        solutions = []
        
        for run in range(num_runs):
            solution, cost, exec_time, total_iterations = tabu_search(
                n, p, w, c, alpha, 
                max_iterations=max_iterations, 
                tabu_tenure=tenure,
                dataset_name=dataset_name
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
        std_cost = (sum((x - avg_cost)**2 for x in costs) / len(costs))**0.5
        avg_time = sum(times) / len(times)
        
        results.append({
            'dataset': dataset_name,
            'p': p,
            'max_iterations': max_iterations,
            'alpha': alpha,
            'tabu_tenure': tenure,
            'best_solution': best_solution,
            'selected_hub_capacities': selected_capacities,
            'best_cost': best_cost,
            'avg_cost': avg_cost,
            'std_cost': std_cost,
            'avg_time': avg_time,
            'avg_total_iterations': sum(total_iterations_list) / len(total_iterations_list),
            'total_iterations_for_best_cost': total_iterations_for_best_cost
        })
        
        print(f"  Best Solution: {best_solution}")
        print(f"  Selected Hub Capacities: {selected_capacities}")
        print(f"  Best Cost: {best_cost:.4f}, Avg Cost: {avg_cost:.4f}, "
              f"Std Dev: {std_cost:.4f}, Avg Time: {avg_time:.4f}s, "
              f"Avg Total Iterations: {sum(total_iterations_list) / len(total_iterations_list):.1f}, "
              f"# of Iterations (best cost): {total_iterations_for_best_cost}")
    
    return pd.DataFrame(results)


def select_best_config_index(results_df):
    """Select best row by best_cost, using avg_cost as tie-breaker."""
    return results_df.sort_values(by=['best_cost', 'avg_cost'], ascending=[True, True]).index[0]


def main():
    """Main function for parameter tuning with fixed parameters"""
    
    # Fixed parameters
    alpha = 0.3
    fixed_max_iterations = {
        'CAB10': 200,
        'CAB20': 500,
        'CAB25': 800,
        'TR40': 1000,
        'TR55': 1000,
        'RGP100': 1000
    }
    
    # Test configurations
    # For p=3: tabu_tenure = [4, 5, 6, 7]
    # For p=4: tabu_tenure = [6, 7, 8, 9]
    
    all_results = []
    
    print("\n" + "="*70)
    print("TABU SEARCH PARAMETER TUNING")
    print("="*70)
    print(f"\nFixed Parameters:")
    print(f"  Alpha: {alpha}")
    print(f"  Max Iterations - CAB10: {fixed_max_iterations['CAB10']}, "
            f"CAB20: {fixed_max_iterations['CAB20']}, "
            f"CAB25: {fixed_max_iterations['CAB25']}, "
            f"TR40: {fixed_max_iterations['TR40']}, "
            f"TR55: {fixed_max_iterations['TR55']}, "
            f"RGP100: {fixed_max_iterations['RGP100']}")
    
    # Test CAB10 with p=3
    results_cab10_p3 = test_fixed_parameters('CAB10', p=3, tabu_tenure_range=[4, 5, 6, 7], num_runs=5)
    all_results.append(results_cab10_p3)
    # save_results(results_cab10_p3, 'tabu_CAB10_p3_results.csv')
    
    # Test CAB10 with p=4
    results_cab10_p4 = test_fixed_parameters('CAB10', p=4, tabu_tenure_range=[6, 7, 8, 9], num_runs=5)
    all_results.append(results_cab10_p4)
    # save_results(results_cab10_p4, 'tabu_CAB10_p4_results.csv')
    
    # Test CAB20 with p=3
    results_cab20_p3 = test_fixed_parameters('CAB20', p=3, tabu_tenure_range=[4, 5, 6, 7], num_runs=5)
    all_results.append(results_cab20_p3)
    # save_results(results_cab20_p3, 'tabu_CAB20_p3_results.csv')
    
    # Test CAB20 with p=5
    results_cab20_p5 = test_fixed_parameters('CAB20', p=5, tabu_tenure_range=[8, 9, 10, 11], num_runs=5)
    all_results.append(results_cab20_p5)
    # save_results(results_cab20_p5, 'tabu_CAB20_p5_results.csv')
    
    # Test CAB25 with p=3
    results_cab25_p3 = test_fixed_parameters('CAB25', p=3, tabu_tenure_range=[5, 6, 7, 8], num_runs=5)
    all_results.append(results_cab25_p3)
    # save_results(results_cab25_p3, 'tabu_CAB25_p3_results.csv')
    
    # Test CAB25 with p=5
    results_cab25_p5 = test_fixed_parameters('CAB25', p=5, tabu_tenure_range=[8, 9, 10, 11], num_runs=5)
    all_results.append(results_cab25_p5)
    # save_results(results_cab25_p5, 'tabu_CAB25_p5_results.csv')

    # Test TR40 with p=4
    results_tr40_p4 = test_fixed_parameters('TR40', p=4, tabu_tenure_range=[7, 8, 9, 10], num_runs=5)
    all_results.append(results_tr40_p4)
    # save_results(results_tr40_p4, 'tabu_TR40_p4_results.csv')
    
    # Test TR40 with p=6
    results_tr40_p6 = test_fixed_parameters('TR40', p=6, tabu_tenure_range=[10, 11, 12], num_runs=5)
    all_results.append(results_tr40_p6)
    # save_results(results_tr40_p6, 'tabu_TR40_p6_results.csv')
    
    # Test TR55 with p=5
    results_tr55_p5 = test_fixed_parameters('TR55', p=5, tabu_tenure_range=[11, 12, 13], num_runs=5)
    all_results.append(results_tr55_p5)
    # save_results(results_tr55_p5, 'tabu_TR55_p5_results.csv')
    
    # Test TR55 with p=7
    results_tr55_p7 = test_fixed_parameters('TR55', p=7, tabu_tenure_range=[12, 13, 14, 15], num_runs=5)
    all_results.append(results_tr55_p7)
    # save_results(results_tr55_p7, 'tabu_TR55_p7_results.csv')
    
    # Test RGP100 with p=9
    results_rgp100_p9 = test_fixed_parameters('RGP100', p=9, tabu_tenure_range=[17, 18, 19, 20], num_runs=5)
    all_results.append(results_rgp100_p9)
    # save_results(results_rgp100_p9, 'tabu_RGP100_p9_results.csv')
    
    # Test RGP100 with p=12
    results_rgp100_p12 = test_fixed_parameters('RGP100', p=12, tabu_tenure_range=[21, 22, 23, 24], num_runs=5)
    all_results.append(results_rgp100_p12)
    # save_results(results_rgp100_p12, 'tabu_RGP100_p12_results.csv')
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    save_results(combined_results, 'Param_tuning_tabu.csv')
    
    # Print summary
    print("\n" + "="*70)
    print("BEST PARAMETERS SUMMARY")
    print("="*70)
    print("Selection rule: lowest best_cost; if tied, lowest avg_cost")
    
    # Best for each dataset-p combination
    summary_p_values = {
        'CAB10': [3, 4],
        'CAB20': [3, 5],
        'CAB25': [3, 5],
        'TR40': [4, 6],
        'TR55': [5, 7],
        'RGP100': [9, 12]
    }

    for dataset in ['CAB10', 'CAB20', 'CAB25', 'TR40', 'TR55', 'RGP100']:
        for p in summary_p_values[dataset]:
            subset = combined_results[(combined_results['dataset'] == dataset) & 
                                     (combined_results['p'] == p)]
            if len(subset) > 0:
                best_idx = select_best_config_index(subset)
                best = combined_results.loc[best_idx]
                print(f"\n{dataset} - p={best['p']}")
                print(f"  Best tabu_tenure: {int(best['tabu_tenure'])}")
                print(f"  Best Solution: {best['best_solution']}")
                if 'selected_hub_capacities' in best:
                    print(f"  Selected Hub Capacities: {best['selected_hub_capacities']}")
                print(f"  Best Cost: {best['best_cost']:.4f}")
                print(f"  Avg Cost: {best['avg_cost']:.4f}")
                print(f"  Std Dev: {best['std_cost']:.4f}")
                print(f"  Avg Time: {best['avg_time']:.4f}s")
                if 'total_iterations_for_best_cost' in best:
                    print(f"  # of Iterations (best cost): {int(best['total_iterations_for_best_cost'])}")
    
    # Overall best across all configurations
    print("\n" + "="*70)
    print("OVERALL BEST CONFIGURATION")
    print("="*70)
    print("Selection rule: lowest best_cost; if tied, lowest avg_cost")
    best_overall_idx = select_best_config_index(combined_results)
    best_overall = combined_results.loc[best_overall_idx]
    print(f"Dataset: {best_overall['dataset']}")
    print(f"p: {int(best_overall['p'])}")
    print(f"max_iterations: {int(best_overall['max_iterations'])}")
    print(f"alpha: {best_overall['alpha']}")
    print(f"tabu_tenure: {int(best_overall['tabu_tenure'])}")
    print(f"Best Solution: {best_overall['best_solution']}")
    if 'selected_hub_capacities' in best_overall:
        print(f"Selected Hub Capacities: {best_overall['selected_hub_capacities']}")
    print(f"Average Cost: {best_overall['avg_cost']:.4f}")
    print(f"Std Dev: {best_overall['std_cost']:.4f}")
    print(f"Avg Time: {best_overall['avg_time']:.4f}s")
    if 'total_iterations_for_best_cost' in best_overall:
        print(f"# of Iterations (best cost): {int(best_overall['total_iterations_for_best_cost'])}")


if __name__ == "__main__":
    main()
