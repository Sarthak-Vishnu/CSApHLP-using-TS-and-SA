"""
Module: algorithms.py
Description: Meta-heuristic algorithms for solving the Hub Location Problem (HLP)

This module contains various optimization algorithms including:
- Tabu Search
- Simulated Annealing
"""

import copy
import math
import time
import numpy as np
from .functions import *
import random


def tabu_search(
    n,
    p,
    w,
    c,
    alpha,
    max_iterations=100,
    tabu_tenure=10,
    dataset_name=None,
    capacities=None,
    fixed_costs=None,
):
    """
    Tabu Search algorithm for HLP with compound stopping rule
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor for inter-hub connections
        max_iterations (int): Maximum number of iterations (N-rule)
        tabu_tenure (int): Number of iterations a move remains tabu
        dataset_name (str|None): Dataset key for capacity profile selection
        capacities (dict|None): Optional explicit capacity profile
        fixed_costs (dict|None): Optional explicit fixed-cost profile
        
    Returns:
        tuple: (best_solution, best_cost, execution_time, total_iterations)
    """
    start_time = time.time()
    
    # # Generate initial solution by random allocation
    # current_solution = initial_solution(n, p)
    
    # Generate initial solution using closest hub allocation
    current_solution = initial_solution_closest(n, p, c)
    current_cost = cost_evaluation(
        current_solution,
        w,
        c,
        alpha,
        dataset_name=dataset_name,
        capacities=capacities,
        fixed_costs=fixed_costs,
    )
    
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    
    # Tabu list stores recently visited solutions
    tabu_list = []
    
    # Compound stopping rule variables
    iteration = 0
    iterations_without_improvement = 0
    max_iterations_without_improvement = int(0.1 * max_iterations)  # W-rule: W = 0.1 × N
    progress_interval = 10
    
    # Continue while both stopping conditions are not met
    while iteration < max_iterations and iterations_without_improvement < max_iterations_without_improvement:
        # Generate neighbors using both NS1 and NS2
        neighbors = []
        
        # Get spokes (non-hub nodes)
        spokes = [i for i in range(1, n+1) if i not in current_solution]
        
        # Generate neighbors using NS1
        for spoke in spokes:
            neighbor = NS1(current_solution, spoke)
            neighbor_cost = cost_evaluation(
                neighbor,
                w,
                c,
                alpha,
                dataset_name=dataset_name,
                capacities=capacities,
                fixed_costs=fixed_costs,
            )
            neighbors.append((neighbor, neighbor_cost))
        
        # Generate neighbors using NS2
        hubs = list(set(current_solution))
        for spoke in spokes:
            for hub in hubs:
                if current_solution[spoke-1] != hub:
                    neighbor = NS2(current_solution, spoke, hub)
                    neighbor_cost = cost_evaluation(
                        neighbor,
                        w,
                        c,
                        alpha,
                        dataset_name=dataset_name,
                        capacities=capacities,
                        fixed_costs=fixed_costs,
                    )
                    neighbors.append((neighbor, neighbor_cost))
        
        # Find best non-tabu neighbor (or best neighbor if aspiration criteria met)
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        for neighbor, neighbor_cost in neighbors:
            # Accept if not in tabu list, or if aspiration criteria met (better than best)
            if neighbor not in tabu_list or neighbor_cost < best_cost:
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
        
        # Move to best neighbor
        if best_neighbor is not None:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            # Update tabu list
            tabu_list.append(copy.deepcopy(current_solution))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            # Update best solution if improved
            if current_cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = current_cost
                iterations_without_improvement = 0  # Reset W-rule counter
            else:
                iterations_without_improvement += 1  # Increment W-rule counter
        
        iteration += 1  # Increment N-rule counter
        if iteration % progress_interval == 0:
            print(f"Tabu Search progress: iteration {iteration}/{max_iterations}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # print(f"Nodes: {n}, Hubs: {p}, Alpha: {alpha}, Max Iterations: {max_iterations}, Tabu Tenure: {tabu_tenure}, Time: {execution_time}")
    
    return best_solution, best_cost, execution_time, iteration


def _generate_random_neighbor(current_solution, n):
    """Generate a random neighbor using NS1/NS2; returns None if no valid move."""
    if random.random() < 0.5:
        # NS1
        spokes = [i for i in range(1, n + 1) if i not in current_solution]
        if not spokes:
            return None
        spoke = random.choice(spokes)
        return NS1(current_solution, spoke)

    # NS2
    hubs = list(set(current_solution))
    spokes = [i for i in range(1, n + 1) if i not in hubs]
    if not spokes:
        return None
    spoke = random.choice(spokes)
    hub_candidates = [h for h in hubs if h != current_solution[spoke - 1]]
    if not hub_candidates:
        return None
    hub = random.choice(hub_candidates)
    return NS2(current_solution, spoke, hub)


def _compute_initial_temperature(
    current_solution,
    w,
    c,
    alpha,
    p0,
    n_samples,
    dataset_name=None,
    capacities=None,
    fixed_costs=None,
):
    """Compute initial temperature based on target acceptance probability."""
    if p0 <= 0 or p0 >= 1:
        raise ValueError("p0 must be between 0 and 1 (exclusive).")

    base_cost = cost_evaluation(
        current_solution,
        w,
        c,
        alpha,
        dataset_name=dataset_name,
        capacities=capacities,
        fixed_costs=fixed_costs,
    )
    delta_sum = 0.0
    count = 0

    for _ in range(n_samples):
        neighbor = _generate_random_neighbor(current_solution, len(current_solution))
        if neighbor is None:
            continue
        delta_cost = (
            cost_evaluation(
                neighbor,
                w,
                c,
                alpha,
                dataset_name=dataset_name,
                capacities=capacities,
                fixed_costs=fixed_costs,
            )
            - base_cost
        )
        if delta_cost > 0:
            delta_sum += delta_cost
            count += 1

    if count == 0:
        delta_avg = 1e-6
    else:
        delta_avg = delta_sum / count

    return -delta_avg / math.log(p0)


def simulated_annealing(n, p, w, c, alpha,
                        initial_temp=None,
                        delta=0.1,
                        max_iterations=1000,
                        p0=0.8,
                        n_samples=50,
                        dataset_name=None,
                        capacities=None,
                        fixed_costs=None):
    """
    Simulated Annealing algorithm for HLP with compound stopping rule
    and non-adaptive temperature schedule
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor
        initial_temp (float|None): Initial temperature. If None, compute using p0 and n_samples.
        delta (float): Empirical distance parameter
        max_iterations (int): Maximum number of iterations (N-rule)
        p0 (float): Target acceptance probability for uphill moves (0 < p0 < 1)
        n_samples (int): Number of random neighbors to estimate initial temperature
        dataset_name (str|None): Dataset key for capacity profile selection
        capacities (dict|None): Optional explicit capacity profile
        fixed_costs (dict|None): Optional explicit fixed-cost profile
        
    Returns:
        tuple: (best_solution, best_cost, execution_time, total_iterations)
    """
    start_time = time.time()
    
    # Generate initial solution using closest hub allocation
    current_solution = initial_solution_closest(n, p, c)
    current_cost = cost_evaluation(
        current_solution,
        w,
        c,
        alpha,
        dataset_name=dataset_name,
        capacities=capacities,
        fixed_costs=fixed_costs,
    )
    
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    
    # Temperature
    if initial_temp is None:
        temperature = _compute_initial_temperature(
            current_solution,
            w,
            c,
            alpha,
            p0,
            n_samples,
            dataset_name=dataset_name,
            capacities=capacities,
            fixed_costs=fixed_costs,
        )
        print(f"SA initial temperature (computed): {temperature:.6f}")
    else:
        temperature = initial_temp

    # Temperature-based stopping rule: Tf = 1e-3 * T0
    initial_temperature = max(float(temperature), 1e-12)
    final_temperature = 1e-3 * initial_temperature
    
    # Compound stopping rule variables
    iteration = 0
    iterations_without_improvement = 0
    max_iterations_without_improvement = int(0.1 * max_iterations)  # W-rule

    # Temperature stage length for sigma_i estimation (objective variations)
    stage_length = max(5, min(25, n))
    
    # Continue while both stopping conditions are not met
    while (
        iteration < max_iterations
        and iterations_without_improvement < max_iterations_without_improvement
        and temperature > final_temperature
    ):
        # Collect objective variations in current temperature stage
        stage_variations = []

        for _ in range(stage_length):
            if (
                iteration >= max_iterations
                or iterations_without_improvement >= max_iterations_without_improvement
            ):
                break

            # Randomly select neighborhood
            neighbor = _generate_random_neighbor(current_solution, n)
            if neighbor is None:
                iteration += 1
                continue

            neighbor_cost = cost_evaluation(
                neighbor,
                w,
                c,
                alpha,
                dataset_name=dataset_name,
                capacities=capacities,
                fixed_costs=fixed_costs,
            )
            delta_cost = neighbor_cost - current_cost
            stage_variations.append(delta_cost)

            # Acceptance criterion
            if delta_cost < 0:
                # Accept improving solution
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                # Probabilistic acceptance
                safe_temperature = max(temperature, 1e-12)
                acceptance_probability = math.exp(-delta_cost / safe_temperature)
                if random.random() < acceptance_probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    iterations_without_improvement += 1
                else:
                    iterations_without_improvement += 1

            iteration += 1

        # Compute sigma_i from objective variations observed in this temperature stage
        if len(stage_variations) > 1:
            sigma_i = float(np.std(stage_variations))
        else:
            sigma_i = 0.0

        # Temperature update (Van Laarhoven & Aarts non-adaptive schedule)
        if sigma_i > 0 and np.isfinite(sigma_i) and temperature > 0:
            denominator = 1 + (temperature * math.log(1 + delta)) / (3 * sigma_i)
            if denominator > 0 and np.isfinite(denominator):
                temperature = temperature / denominator
            else:
                break
        else:
            # If there is no variation, search is effectively frozen at this stage
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_solution, best_cost, execution_time, iteration


