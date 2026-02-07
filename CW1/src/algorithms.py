"""
Module: algorithms.py
Description: Meta-heuristic algorithms for solving the Hub Location Problem (HLP)

This module contains various optimization algorithms including:
- Variable Neighborhood Descent (VND)
- Tabu Search
- Simulated Annealing
- Genetic Algorithm
"""

import copy
import math
import time
import numpy as np
from .functions import *
import random

'''
def VND(n, p, w, c, alpha, k_max=2):
    """
    Variable Neighborhood Descent algorithm for HLP
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor for inter-hub connections
        k_max (int): Maximum number of neighborhood structures (default: 2)
        
    Returns:
        tuple: (best_solution, best_cost)
    """
    # Generate initial solution
    current_solution = initial_solution(n, p)
    current_cost = cost_evaluation(current_solution, w, c, alpha)
    
    k = 1  # Start with first neighborhood structure
    
    while k <= k_max:
        # Perform local search with neighborhood structure k
        if k == 1:
            neighbor_solution, neighbor_cost = LS_NS1_Steepest(current_solution, w, c, alpha)
        elif k == 2:
            neighbor_solution, neighbor_cost = LS_NS2_Steepest(current_solution, w, c, alpha)
        
        # If improvement found, accept and restart from first neighborhood
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            k = 1  # Restart from first neighborhood
        else:
            k += 1  # Move to next neighborhood
    
    return current_solution, current_cost


def VND_with_replications(n, p, w, c, alpha, k_max=2, replications=20, verbose=False):
    """
    Run VND multiple times and return the best solution
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor for inter-hub connections
        k_max (int): Maximum number of neighborhood structures
        replications (int): Number of times to run VND
        verbose (bool): Print progress for each replication
        
    Returns:
        tuple: (best_solution, best_cost, all_costs)
    """
    best_solution = None
    best_cost = float('inf')
    all_costs = []
    
    for rep in range(replications):
        if verbose:
            print(f"\n----- Replication {rep + 1}/{replications} -----")
        
        solution, cost = VND(n, p, w, c, alpha, k_max)
        all_costs.append(cost)
        
        if verbose:
            print(f"Solution: {solution}")
            print(f"Cost: {cost:.4f}")
        
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    return best_solution, best_cost, all_costs
'''

def tabu_search(n, p, w, c, alpha, max_iterations=100, tabu_tenure=10):
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
        
    Returns:
        tuple: (best_solution, best_cost, execution_time, total_iterations)
    """
    start_time = time.time()
    
    # # Generate initial solution by random allocation
    # current_solution = initial_solution(n, p)
    
    # Generate initial solution using closest hub allocation
    current_solution = initial_solution_closest(n, p, c)
    current_cost = cost_evaluation(current_solution, w, c, alpha)
    
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
            neighbor_cost = cost_evaluation(neighbor, w, c, alpha)
            neighbors.append((neighbor, neighbor_cost))
        
        # Generate neighbors using NS2
        hubs = list(set(current_solution))
        for spoke in spokes:
            for hub in hubs:
                if current_solution[spoke-1] != hub:
                    neighbor = NS2(current_solution, spoke, hub)
                    neighbor_cost = cost_evaluation(neighbor, w, c, alpha)
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

'''
def simulated_annealing(n, p, w, c, alpha, initial_temp=1000, cooling_rate=0.95, 
                        max_iterations=1000):
    """
    Simulated Annealing algorithm for HLP
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor for inter-hub connections
        initial_temp (float): Initial temperature
        cooling_rate (float): Temperature cooling rate (0 < rate < 1)
        max_iterations (int): Maximum number of iterations
        
    Returns:
        tuple: (best_solution, best_cost, execution_time, total_iterations)
    """
    start_time = time.time()
    
    # Generate initial solution
    current_solution = initial_solution(n, p)
    current_cost = cost_evaluation(current_solution, w, c, alpha)
    
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate a random neighbor
        if random.random() < 0.5:
            # Use NS1
            spokes = [i for i in range(1, n+1) if i not in current_solution]
            if spokes:
                spoke = random.choice(spokes)
                neighbor = NS1(current_solution, spoke)
        else:
            # Use NS2
            hubs = list(set(current_solution))
            spokes = [i for i in range(1, n+1) if i not in hubs]
            if spokes:
                spoke = random.choice(spokes)
                hub_candidates = [h for h in hubs if h != current_solution[spoke-1]]
                if hub_candidates:
                    hub = random.choice(hub_candidates)
                    neighbor = NS2(current_solution, spoke, hub)
                else:
                    continue
            else:
                continue
        
        neighbor_cost = cost_evaluation(neighbor, w, c, alpha)
        
        # Calculate cost difference
        delta = neighbor_cost - current_cost
        
        # Accept or reject the neighbor
        if delta < 0:  # Better solution
            current_solution = neighbor
            current_cost = neighbor_cost
            
            # Update best solution
            if current_cost < best_cost:
                best_solution = copy.deepcopy(current_solution)
                best_cost = current_cost
        else:  # Worse solution - accept with probability
            acceptance_probability = math.exp(-delta / temperature)
            if random.random() < acceptance_probability:
                current_solution = neighbor
                current_cost = neighbor_cost
        
        # Cool down temperature
        temperature *= cooling_rate
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_solution, best_cost, execution_time, max_iterations
'''



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


def _compute_initial_temperature(current_solution, w, c, alpha, p0, n_samples):
    """Compute initial temperature based on target acceptance probability."""
    if p0 <= 0 or p0 >= 1:
        raise ValueError("p0 must be between 0 and 1 (exclusive).")

    base_cost = cost_evaluation(current_solution, w, c, alpha)
    delta_sum = 0.0
    count = 0

    for _ in range(n_samples):
        neighbor = _generate_random_neighbor(current_solution, len(current_solution))
        if neighbor is None:
            continue
        delta_cost = cost_evaluation(neighbor, w, c, alpha) - base_cost
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
                        n_samples=50):
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
        
    Returns:
        tuple: (best_solution, best_cost, execution_time, total_iterations)
    """
    start_time = time.time()
    
    # Generate initial solution using closest hub allocation
    current_solution = initial_solution_closest(n, p, c)
    current_cost = cost_evaluation(current_solution, w, c, alpha)
    
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    
    # Temperature
    if initial_temp is None:
        temperature = _compute_initial_temperature(current_solution, w, c, alpha, p0, n_samples)
    else:
        temperature = initial_temp
    
    # Cost history for sigma_i computation
    cost_history = [current_cost]
    
    # Compound stopping rule variables
    iteration = 0
    iterations_without_improvement = 0
    max_iterations_without_improvement = int(0.1 * max_iterations)  # W-rule
    
    # Continue while both stopping conditions are not met
    while iteration < max_iterations and iterations_without_improvement < max_iterations_without_improvement:
        
        # Randomly select neighborhood
        neighbor = _generate_random_neighbor(current_solution, n)
        if neighbor is None:
            iteration += 1
            continue
        
        neighbor_cost = cost_evaluation(neighbor, w, c, alpha)
        delta_cost = neighbor_cost - current_cost
        
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
            acceptance_probability = math.exp(-delta_cost / temperature)
            if random.random() < acceptance_probability:
                current_solution = neighbor
                current_cost = neighbor_cost
                iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
        
        # Update cost history
        cost_history.append(current_cost)
        
        # Compute sigma_i (standard deviation of visited costs)
        if len(cost_history) > 1:
            sigma_i = np.std(cost_history)
        else:
            sigma_i = 0
        
        # Temperature update (non-adaptive schedule from slide)
        if sigma_i > 0:
            temperature = temperature / (
                1 + (temperature * math.log(1 + delta)) / (3 * sigma_i)
            )
        
        iteration += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return best_solution, best_cost, execution_time, iteration


'''
def multi_start_local_search(n, p, w, c, alpha, num_starts=20, neighborhood='both'):
    """
    Multi-start local search algorithm
    
    Args:
        n (int): Number of nodes
        p (int): Number of hubs
        w (DataFrame): Flow/demand matrix
        c (DataFrame): Cost/distance matrix
        alpha (float): Discount factor for inter-hub connections
        num_starts (int): Number of random starts
        neighborhood (str): 'NS1', 'NS2', or 'both'
        
    Returns:
        tuple: (best_solution, best_cost, all_costs)
    """
    best_solution = None
    best_cost = float('inf')
    all_costs = []
    
    for start in range(num_starts):
        # Generate random initial solution
        current_solution = initial_solution(n, p)
        improved = True
        
        # Local search until no improvement
        while improved:
            improved = False
            
            if neighborhood in ['NS1', 'both']:
                neighbor_solution, neighbor_cost = LS_NS1_Steepest(current_solution, w, c, alpha)
                if neighbor_cost < cost_evaluation(current_solution, w, c, alpha):
                    current_solution = neighbor_solution
                    improved = True
            
            if neighborhood in ['NS2', 'both']:
                neighbor_solution, neighbor_cost = LS_NS2_Steepest(current_solution, w, c, alpha)
                if neighbor_cost < cost_evaluation(current_solution, w, c, alpha):
                    current_solution = neighbor_solution
                    improved = True
        
        current_cost = cost_evaluation(current_solution, w, c, alpha)
        all_costs.append(current_cost)
        
        if current_cost < best_cost:
            best_solution = copy.deepcopy(current_solution)
            best_cost = current_cost
    
    return best_solution, best_cost, all_costs
'''