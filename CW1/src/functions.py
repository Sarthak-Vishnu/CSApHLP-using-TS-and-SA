
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

def cost_evaluation(array, w, c, alpha):
    """Vectorized cost evaluation.

    This keeps the same formula as the nested loops but uses NumPy for speed.
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

    cost = (w_np * (term_origin[:, None] + alpha * hub_to_hub + term_dest[None, :])).sum()
    return float(cost)

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