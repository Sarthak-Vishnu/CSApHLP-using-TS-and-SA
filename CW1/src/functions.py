
"""
Module: functions.py
Description: Hub location problem (HLP) utilities for solving facility location optimization.
Functions:
    initial_solution(n, p): Generate an initial solution by randomly allocating n nodes to p hubs.
        Args:
            n (int): Total number of nodes
            p (int): Number of hubs to select
        Returns:
            list: Array where array[i] contains the hub assignment for node i+1
    
    initial_solution_closest(n, p, c): Generate an initial solution by allocating nodes to closest hub.
        Args:
            n (int): Total number of nodes
            p (int): Number of hubs to select
            c (pd.DataFrame): Cost/distance matrix
        Returns:
            list: Array where array[i] contains the hub assignment for node i+1
    
    hlp_graph(array): Visualize the hub location problem solution as a graph.
        Args:
            array (list): Hub allocation array where array[i] is the hub for node i+1
        Returns:
            matplotlib.figure.Figure: Figure object containing the visualization

Global Variables:
    x_coor (list): X-coordinates for 10 nodes used in graph visualization
    y_coor (list): Y-coordinates for 10 nodes used in graph visualization
    labels (list): Node labels (1-10) used in graph annotation
Note:
    x_coor and y_coor are module-level constants that store the spatial coordinates of nodes.
    They are used by hlp_graph() to position nodes on a 2D scatter plot and draw connections
    between hubs (red lines) and between spokes and their assigned hubs (black dashed lines).
"""

'''
Improvements (to be implemented)
0. Implement final temperature as a part of stopping condition
1. Write main() with required param as assign. brief's table
2. [CAB✅, TR✅done i.e. code, RGP✅done i.e. code] Experiment with Tabu_tenure values and max_iter for each dataset.
3. ✅Implement algo for TR40, TR55, and RGP100
4. 


Completed:
> ✅ Change stopping condition in Tabu search (compound stopping strategy). 
> ✅ Set 'max_iter' and 'tabu_tenure' as args during function call for the algorithm
> ✅ Change 'initial solution generation' function to allocate the rest of the nodes to their closest hub.
> ✅ [2 are enough] Implement other neighbourhood generation functions (all 4 from WS1 slides)
> ✅ add exec_time as return arg for Tabu Search algo
> ✅ add '# of iterations' of completion of algo, as return arg for Tabu Search algo
> Implement [Tabu Search ✅] and [Simulated Annealing  ✅]


'''
import os
import numpy as np
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt

import random
from random import *

import warnings
warnings.filterwarnings("ignore")


FIXED_COSTS_BY_LEVEL = {
    "L1": 50_000_000,
    "L2": 100_000_000,
    "L3": 150_000_000,
}

CAB10_CAPACITIES = {"L1": 199805.2, "L2": 399610.4, "L3": 849172.1}
CAB20_CAPACITIES = {"L1": 1150918.8, "L2": 2301837.6, "L3": 4891404.9}
CAB25_CAPACITIES = {"L1": 1708001.2, "L2": 3416002.4, "L3": 7259005.1}
TR40_CAPACITIES = {"L1": 3157851.46, "L2": 6315702.92, "L3": 13420868.71}
TR55_CAPACITIES = {"L1": 8150485.4, "L2": 16300970.8, "L3": 34639562.95}
RGP100_CAPACITIES = {"L1": 19800492.8, "L2": 39600985.6, "L3": 84152094.4}

DATASET_CAPACITIES_BY_NAME = {
    "CAB10": CAB10_CAPACITIES,
    "CAB20": CAB20_CAPACITIES,
    "CAB25": CAB25_CAPACITIES,
    "TR40": TR40_CAPACITIES,
    "TR55": TR55_CAPACITIES,
    "RGP100": RGP100_CAPACITIES,
}

DATASET_CAPACITIES_BY_NODES = {
    10: CAB10_CAPACITIES,
    20: CAB20_CAPACITIES,
    25: CAB25_CAPACITIES,
    40: TR40_CAPACITIES,
    55: TR55_CAPACITIES,
    100: RGP100_CAPACITIES,
}


def _normalize_dataset_name(dataset_name):
    if dataset_name is None:
        return None
    return str(dataset_name).strip().replace(" ", "").upper()


def _get_capacity_levels(dataset_name, n, capacities):
    if capacities is not None:
        return capacities

    normalized_name = _normalize_dataset_name(dataset_name)
    if normalized_name in DATASET_CAPACITIES_BY_NAME:
        return DATASET_CAPACITIES_BY_NAME[normalized_name]

    return DATASET_CAPACITIES_BY_NODES.get(n)


def _auto_scale_capacities_for_normalized_flow(capacity_levels, total_flow):
    """Scale capacities when flow matrix has been normalized to sum ~1.

    Dataset capacities are defined in original flow units. If ``w`` is normalized,
    hub loads are also normalized and capacities must be scaled consistently.
    """
    if not capacity_levels:
        return capacity_levels

    if not np.isclose(total_flow, 1.0, rtol=1e-4, atol=1e-6):
        return capacity_levels

    # If capacities are already normalized-scale (<= 1), keep them as-is.
    if max(capacity_levels.values()) <= 1.0:
        return capacity_levels

    inferred_total_flow = _infer_total_flow_from_capacity_levels(capacity_levels)
    if inferred_total_flow <= 0:
        return capacity_levels

    return {level: float(value) / inferred_total_flow for level, value in capacity_levels.items()}


def _infer_total_flow_from_capacity_levels(capacity_levels):
    """Infer original total flow from capacity policy ratios.

    Assumes policy levels are set as fractions of total flow:
    L1 = 0.2F, L2 = 0.4F, L3 = 0.85F.
    """
    if not capacity_levels:
        return 0.0

    inferred_totals = []
    if capacity_levels.get("L1", 0) > 0:
        inferred_totals.append(float(capacity_levels["L1"]) / 0.2)
    if capacity_levels.get("L2", 0) > 0:
        inferred_totals.append(float(capacity_levels["L2"]) / 0.4)
    if capacity_levels.get("L3", 0) > 0:
        inferred_totals.append(float(capacity_levels["L3"]) / 0.85)

    if not inferred_totals:
        return 0.0

    return float(np.mean(inferred_totals))


def _selected_hub_capacities_from_precomputed(assignments, incoming_per_node, capacity_levels, return_loads=False):
    """Fast hub-capacity classification using precomputed arrays.

    Args:
        assignments (np.ndarray): 1-based hub assignment vector of length n.
        incoming_per_node (np.ndarray): Incoming flow per node of length n.
        capacity_levels (dict): Capacity thresholds with keys L1/L2/L3.
        return_loads (bool): If True, include per-hub load dictionary.
    """
    hubs_in_solution_order = list(dict.fromkeys(assignments.tolist()))

    # Aggregate load per hub index in O(n)
    hub_load_by_label = np.bincount(assignments, weights=incoming_per_node, minlength=assignments.size + 1)

    l1 = float(capacity_levels["L1"])
    l2 = float(capacity_levels["L2"])
    l3 = float(capacity_levels["L3"])

    hub_levels = {}
    hub_loads = {}

    for hub in hubs_in_solution_order:
        hub_int = int(hub)
        hub_load = float(hub_load_by_label[hub_int])
        hub_loads[hub_int] = hub_load

        if hub_load <= l1:
            hub_levels[hub_int] = "L1"
        elif hub_load <= l2:
            hub_levels[hub_int] = "L2"
        elif hub_load <= l3:
            hub_levels[hub_int] = "L3"
        else:
            if return_loads:
                return {}, True, hub_loads
            return {}, True

    if return_loads:
        return hub_levels, False, hub_loads
    return hub_levels, False


def selected_hub_capacities(array, w, dataset_name=None, capacities=None, return_loads=False):
    """Return selected capacity level for each open hub in solution order.

    Args:
        array (list[int]): Hub assignment; array[i] = assigned hub for node i+1.
        w (pd.DataFrame|np.ndarray): Flow matrix.
        dataset_name (str|None): Dataset key to select predefined capacities.
        capacities (dict|None): Optional explicit capacities dict {"L1","L2","L3"}.
        return_loads (bool): If True, also return hub loads.

    Returns:
        dict | tuple:
            - hub_levels: {hub: "Lx"} in first-appearance hub order.
            - infeasible: bool
            - hub_loads (optional): {hub: load}
    """
    w_np = w.to_numpy() if hasattr(w, "to_numpy") else np.asarray(w)
    n = len(array)

    capacity_levels = _get_capacity_levels(dataset_name, n, capacities)
    if capacity_levels is None:
        result = ({}, False, {}) if return_loads else ({}, False)
        return result

    capacity_levels = _auto_scale_capacities_for_normalized_flow(capacity_levels, float(w_np.sum()))

    assignments = np.asarray(array, dtype=np.int64)
    incoming_per_node = w_np.sum(axis=0)
    return _selected_hub_capacities_from_precomputed(
        assignments,
        incoming_per_node,
        capacity_levels,
        return_loads=return_loads,
    )



# FUNCTION: Generate initial solution by random allocation (n: total nodes, p: hubs)
def initial_solution(n, p):
    array = [None]*n            # Create an empty array/list of size n
    
    # Randomly select p disting hubs from n nodes
    hubs = sample (range(1, n+1), p)
    # print(hubs)
    
    # Allocate hubs to themselves
    for i in range(len(hubs)):
        array[hubs[i]-1] = hubs[i];
        
    # Allocate remaining non-hub nodes to a hub, randomly
    for i in range(len(array)):
        if array[i] == None:
            array[i] = choice(hubs)
            
    return array


# FUNCTION: Generate initial solution by allocating nodes to closest hub
def initial_solution_closest(n, p, c):
    """
    Generate an initial solution by allocating each node to its closest hub
    
    Args:
        n (int): Total number of nodes
        p (int): Number of hubs to select
        c (pd.DataFrame): Cost/distance matrix
        
    Returns:
        list: Array where array[i] contains the hub assignment for node i+1
    """
    array = [None]*n            # Create an empty array/list of size n
    
    # Randomly select p distinct hubs from n nodes
    hubs = sample(range(1, n+1), p)
    
    # Allocate hubs to themselves
    for hub in hubs:
        array[hub-1] = hub
        
    # Allocate remaining non-hub nodes to their closest hub based on cost matrix
    for i in range(n):
        if array[i] is None:  # If node i+1 is not a hub
            min_cost = float('inf')
            closest_hub = None
            
            # Find the closest hub
            for hub in hubs:
                cost = c.iloc[i, hub-1]  # Cost from node i+1 to hub
                if cost < min_cost:
                    min_cost = cost
                    closest_hub = hub
            
            array[i] = closest_hub
            
    return array



x_coor = sample(range(0,10), 10)
y_coor = sample(range(0,10), 10)
labels = [i for i in range(1, 11)]
def hlp_graph(array):
    '''
    FUNCTION: Visualize the hub location problem solution as a graph 
    Args:
        array (list): Hub allocation array where array[i] is the hub for node i+1
    Returns:
        matplotlib.figure.Figure: Figure object containing the visualization    
    '''
    
    an = len(array)
    
    # x_coor = np.random.uniform(0, 10, an)  # Continuous values in [0, 10]
    # y_coor = np.random.uniform(0, 10, an)
    # labels = [i for i in range(1, an + 1)]
    
    colors = ["k"]*an
    shape = ["o"]*an
    size = [20]*an
    
    fig, ax = plt.subplots()
    
    for i in range(len(array)):
        if array[i] == i+1:
            colors[i] = "r"
            shape[i] = "s"
            size[i] = 100
            
    for _m, _c, _s, _x, _y in zip(shape, colors, size, x_coor, y_coor):
        ax.scatter(_x, _y, s=_s, marker=_m, c=_c)
        
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_coor[i]+0.2, y_coor[i]+0.2), size=8)
    
    hubs_m = list(set(array))    
    spokes_m = [i for i in range(1, len(array)+1) if i not in hubs_m]
    
    for m in hubs_m:
        for k in hubs_m:
            ax.plot([x_coor[m-1], x_coor[k-1]], [y_coor[m-1], y_coor[k-1]], "r-")
            
    for i in spokes_m:
        ax.plot([x_coor[i-1], x_coor[array[i-1]-1]], [y_coor[i-1], y_coor[array[i-1]-1]], "k--")
        
    # plt.close(fig)
    return fig          # just return, do NOT call plt.show()

def cost_evaluation(array, w, c, alpha, dataset_name=None, capacities=None, fixed_costs=None, return_breakdown=False):
    """Vectorized cost evaluation for capacitated HLP.

    Total Cost = Network Cost + Setup Cost

    - Network cost follows the existing formulation.
    - Setup cost is computed by classifying each open hub into L1/L2/L3 based on
      incoming load generated by nodes assigned to that hub.
    - The smallest capacity level that can handle the load is selected.
    - If no level can handle a hub's load, the solution is infeasible and returns ``inf``.

    Args:
        array (list[int]): Hub assignment; array[i] = assigned hub for node i+1.
        w (pd.DataFrame|np.ndarray): Flow matrix.
        c (pd.DataFrame|np.ndarray): Cost matrix.
        alpha (float): Inter-hub discount factor.
        dataset_name (str|None): Optional dataset key (CAB10, CAB20, CAB25, TR40,
            TR55, RGP100). If provided, capacities are taken from
            ``DATASET_CAPACITIES_BY_NAME`` unless ``capacities`` is explicitly given.
        capacities (dict|None): Optional capacity dict {"L1": x, "L2": y, "L3": z}.
            If None, inferred from ``dataset_name`` or fallback to dataset size.
            If flow matrix ``w`` is normalized (sum ~= 1), capacity values are
            automatically scaled to normalized units for feasibility checks.
        fixed_costs (dict|None): Optional fixed-cost dict by level.
            If None, uses ``FIXED_COSTS_BY_LEVEL``.
        return_breakdown (bool): If True, return (total, network, setup, hub_levels).

    Returns:
        float | tuple: Total cost, or detailed tuple when return_breakdown=True.
    """
    # Convert to NumPy arrays (handles pandas DataFrame or ndarray inputs)
    w_np = w.to_numpy() if hasattr(w, "to_numpy") else np.asarray(w)
    c_np = c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)

    hubs = np.asarray(array, dtype=np.int64) - 1
    n = hubs.size
    idx = np.arange(n)

    term_origin = c_np[idx, hubs]  # c[i, hub_i]
    term_dest = c_np[hubs, idx]    # c[hub_j, j] for each j
    hub_to_hub = c_np[np.ix_(hubs, hubs)]  # c[hub_i, hub_j]

    total_flow = float(w_np.sum())
    network_cost = float((w_np * (term_origin[:, None] + alpha * hub_to_hub + term_dest[None, :])).sum())

    capacity_levels = _get_capacity_levels(dataset_name, n, capacities)
    if capacity_levels is None:
        if return_breakdown:
            return network_cost, network_cost, 0.0, {}
        return network_cost

    # Keep units consistent: if w is normalized but capacities are in original units,
    # rescale network cost back to original flow units.
    if np.isclose(total_flow, 1.0, rtol=1e-4, atol=1e-6) and max(capacity_levels.values()) > 1.0:
        inferred_total_flow = _infer_total_flow_from_capacity_levels(capacity_levels)
        if inferred_total_flow > 0:
            network_cost *= inferred_total_flow

    capacity_levels = _auto_scale_capacities_for_normalized_flow(capacity_levels, total_flow)

    fixed_cost_levels = fixed_costs or FIXED_COSTS_BY_LEVEL

    assignments = np.asarray(array, dtype=np.int64)
    incoming_per_node = w_np.sum(axis=0)
    hub_levels, infeasible = _selected_hub_capacities_from_precomputed(
        assignments,
        incoming_per_node,
        capacity_levels,
        return_loads=False,
    )

    if infeasible:
        if return_breakdown:
            return float("inf"), network_cost, float("inf"), {}
        return float("inf")

    setup_cost = 0.0
    for _, selected_level in hub_levels.items():
        setup_cost += float(fixed_cost_levels[selected_level])

    total_cost = network_cost + setup_cost
    if return_breakdown:
        return total_cost, network_cost, setup_cost, hub_levels
    return total_cost

def normalize_flow(w):
    """
    Scale the flow matrix so that total demand equals 1.
    Args:
        w (pd.DataFrame): Flow matrix
    Returns:
        pd.DataFrame: Normalized flow matrix
    """
    w_np = w.to_numpy() if hasattr(w, "to_numpy") else np.asarray(w)
    total_flow = w_np.sum()
    return w / total_flow

# FUNCTION: Neighborhood Structure 1
# Replace a hub with one of the node allocated to this hub
def NS1(array, spoke):

    n_array = copy.deepcopy(array) 
    hub = n_array[spoke-1]  

    for i in range(len(n_array)):  
        if n_array[i] == hub:  
            n_array[i] = spoke  

    return n_array

# FUNCTION: Neighborhood Structure 2 - Re-allocate a non hub node
def NS2(array, spoke, hub):

    n_array = copy.deepcopy(array)
    n_array[spoke-1] = hub

    return n_array  

# Classical Local Search: Steepest Descent
def LS_NS1_Steepest(array, w, c, alpha):

    best_neighbor = array
    best_neighbor_cost = cost_evaluation(array, w, c, alpha)

    spokes = [i for i in range(1, len(array)+1) if i not in array]

    for s in spokes:
        neighbor = NS1(array, s)
        neighbor_cost = cost_evaluation(neighbor, w, c, alpha)

        if neighbor_cost < best_neighbor_cost:
            best_neighbor = neighbor
            best_neighbor_cost = neighbor_cost

    return best_neighbor, best_neighbor_cost

def LS_NS2_Steepest(array, w, c, alpha):

    best_neighbor = array
    best_neighbor_cost = cost_evaluation(array, w, c, alpha)
    
    hubs = list(set(array))
    spokes = [i for i in range(1, len(array)+1) if i not in hubs]

    for s in spokes:
        for h in hubs:
            if array[s-1] != h:
                neighbor = NS2(array, s, h)
                neighbor_cost = cost_evaluation(neighbor, w, c, alpha)

                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

    return best_neighbor, best_neighbor_cost


###############################################################################


# def add(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b

# def multiply(a, b):
#     return a * b

# def divide(a, b):
#     if b == 0:
#         raise ValueError("Cannot divide by zero.")
#     return a / b

# def square(a):
#     return a * a

# def factorial(n):
#     if n < 0:
#         raise ValueError("Factorial is not defined for negative numbers.")
#     if n == 0 or n == 1:
#         return 1
#     result = 1
#     for i in range(2, n + 1):
#         result *= i
#     return result