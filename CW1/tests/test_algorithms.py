import pytest
import pandas as pd
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithms import *
from src.functions import *

@pytest.fixture
def cab10_data():
    """Load CAB10 dataset for testing"""
    
    # Get the directory where main.py is located (src folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to CW1 folder, then into data folder
    dir_name = os.path.join(os.path.dirname(current_dir), 'data')
    file_name = 'CAB_TR_and RGP Datasets_2026'
    file_path = os.path.join(dir_name, file_name + '.xlsx')
    
    print(f"\nLooking for file at: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    w_cab10 = pd.read_excel(
        file_path,
        sheet_name="CAB 10, 20 and 25Nodes",
        header=None,
        usecols="B:K",
        skiprows=2,
        nrows=10
    )
    w_cab10 = normalize_flow(w_cab10)
    
    c_cab10 = pd.read_excel(
        file_path,
        sheet_name="CAB 10, 20 and 25Nodes",
        header=None,
        usecols="B:K",
        skiprows=18,
        nrows=10
    )
    
    return w_cab10, c_cab10


def test_tabu_search_returns_tuple(cab10_data):
    """Test that tabu_search returns a tuple with solution and cost"""
    w, c = cab10_data
    solution, cost, _, _ = tabu_search(n=10, p=3, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert isinstance(solution, list)
    assert isinstance(cost, (int, float))
    assert len(solution) == 10


def test_tabu_search_solution_valid(cab10_data):
    """Test that tabu_search returns a valid solution with correct number of hubs"""
    w, c = cab10_data
    p = 3
    solution, cost, _, _ = tabu_search(n=10, p=p, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert len(set(solution)) == p
    assert all(1 <= hub <= 10 for hub in solution)


def test_tabu_search_cost_positive(cab10_data):
    """Test that tabu_search returns a positive cost"""
    w, c = cab10_data
    solution, cost, _, _ = tabu_search(n=10, p=3, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert cost > 0


def test_tabu_search_cost_matches_evaluation(cab10_data):
    """Test that returned cost matches cost_evaluation function"""
    w, c = cab10_data
    solution, returned_cost, _, _ = tabu_search(n=10, p=3, w=w, c=c, alpha=0.7, max_iterations=10)
    evaluated_cost = cost_evaluation(solution, w, c, 0.7)
    
    assert abs(returned_cost - evaluated_cost) < 1e-6


def test_tabu_search_different_alphas(cab10_data):
    """Test tabu_search with different alpha values"""
    w, c = cab10_data
    alpha_values = [0.3, 0.5, 0.7]
    
    for alpha in alpha_values:
        solution, cost, _, _ = tabu_search(n=10, p=3, w=w, c=c, alpha=alpha, max_iterations=10)
        assert solution is not None
        assert cost > 0


def test_tabu_search_different_p_values(cab10_data):
    """Test tabu_search with different number of hubs"""
    w, c = cab10_data
    p_values = [2, 3, 5]
    
    for p in p_values:
        solution, cost, _, _ = tabu_search(n=10, p=p, w=w, c=c, alpha=0.3, max_iterations=10)
        assert len(set(solution)) == p
        assert cost > 0


def test_tabu_search_tabu_tenure_effect(cab10_data):
    """Test that different tabu_tenure values produce solutions"""
    w, c = cab10_data
    tabu_tenures = [2, 3, 4]
    
    for tenure in tabu_tenures:
        solution, cost, _, _ = tabu_search(n=10, p=3, w=w, c=c, alpha=0.3, 
                           max_iterations=10, tabu_tenure=tenure)
        assert solution is not None
        assert cost > 0


# ==================== SIMULATED ANNEALING TESTS ====================

def test_simulated_annealing_returns_tuple(cab10_data):
    """Test that simulated_annealing returns a tuple with solution, cost, time, and iterations"""
    w, c = cab10_data
    solution, cost, exec_time, total_iterations = simulated_annealing(n=10, p=3, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert isinstance(solution, list)
    assert isinstance(cost, (int, float))
    assert isinstance(exec_time, float)
    assert isinstance(total_iterations, int)
    assert len(solution) == 10


def test_simulated_annealing_solution_valid(cab10_data):
    """Test that simulated_annealing returns a valid solution with correct number of hubs"""
    w, c = cab10_data
    p = 3
    solution, cost, _, _ = simulated_annealing(n=10, p=p, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert len(set(solution)) == p
    assert all(1 <= hub <= 10 for hub in solution)


def test_simulated_annealing_cost_positive(cab10_data):
    """Test that simulated_annealing returns a positive cost"""
    w, c = cab10_data
    solution, cost, _, _ = simulated_annealing(n=10, p=3, w=w, c=c, alpha=0.3, max_iterations=10)
    
    assert cost > 0


def test_simulated_annealing_cost_matches_evaluation(cab10_data):
    """Test that returned cost matches cost_evaluation function"""
    w, c = cab10_data
    solution, returned_cost, _, _ = simulated_annealing(n=10, p=3, w=w, c=c, alpha=0.7, max_iterations=10)
    evaluated_cost = cost_evaluation(solution, w, c, 0.7)
    
    assert abs(returned_cost - evaluated_cost) < 1e-6


def test_simulated_annealing_different_p_values(cab10_data):
    """Test simulated_annealing with different number of hubs"""
    w, c = cab10_data
    p_values = [2, 3, 5]
    
    for p in p_values:
        solution, cost, _, _ = simulated_annealing(n=10, p=p, w=w, c=c, alpha=0.3, max_iterations=10)
        assert len(set(solution)) == p
        assert cost > 0
        

def test_simulated_annealing_different_temperatures(cab10_data):
    """Test simulated_annealing with different initial temperatures"""
    w, c = cab10_data
    temperatures = [500, 1000]
    
    for temp in temperatures:
        solution, cost, _, _ = simulated_annealing(n=10, p=3, w=w, c=c, alpha=0.3, 
                                                    initial_temp=temp, max_iterations=10)
        assert solution is not None
        assert cost > 0



