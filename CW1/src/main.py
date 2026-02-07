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

from functions import *  



def main():
    
    # Get the directory where main.py is located (src folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to CW1 folder, then into data folder
    dir_name = os.path.join(os.path.dirname(current_dir), 'data')
    file_name = 'CAB_TR_and RGP Datasets_2026'
    file_path = os.path.join(dir_name, file_name + '.xlsx')



    # load demand data (flow matrix)
    w_cab10 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,              
        usecols = "B:K",            
        skiprows = 2,              
        nrows = 10                 
    )
    w_cab10 = normalize_flow(w_cab10)
    
    # load unit cost data (cost matrix)
    c_cab10 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,              
        usecols = "B:K",           
        skiprows = 18,               
        nrows = 10                  
    )

    
    # ###### My OLD CODE ########
    # ###### FOR DEBUGGING ######
    # example = initial_solution(10, 3)
    # print(f"\nInitial Solution example: {example}")
    # print('Initial Solution Cost:', cost_evaluation(example, w_cab10, c_cab10, 0.2))
    # # ex_fig = hlp_graph(example)
    
    
    # d = [4, 4, 6, 4, 4, 6, 7, 7, 4, 7]
    # cost_temp = cost_evaluation(d, w_cab10, c_cab10, 0.2)
    # print(f"d = {d}\ncost = {cost_temp}")
    
    # print("\nLocal Search Steppest Descent with NS1 (alpha):")
    # example_11 = LS_NS1_Steepest(example, w_cab10, c_cab10, 0.2)
    # print('Solution:', example_11[0])
    # print('Cost:', example_11[1])
    # # hlp_graph(example_11[0])
    
    # # print("w_cab10:\n", w_cab10)
    # # print("w_cab10 first column:\n", w_cab10.iloc[:, 0])
    # ###### My OLD CODE ########
    
    

    p = [3, 5]
    alpha = [0.3, 0.7]
    
    # ==============================================================================
    print("\n" + "="*60)
    print(" "*22 + "Dataset = CAB10")
    print("="*60)
    
    # Loop through each combination of p and alpha
    for p_val in p:
        for alpha_val in alpha:
            
            print("\n" + "="*60)
            print(f"Running with p={p_val} hubs and alpha={alpha_val}")
            print("="*60)
            
            # Generate initial solution
            example = initial_solution(10, p_val)
            print(f"\nInitial Solution: {example}")
            initial_cost = cost_evaluation(example, w_cab10, c_cab10, alpha_val)
            print(f"Initial Cost: {initial_cost:.4f}\n")
            
            # COPY THE BELOW PART FOR ADDING ANOTHER ALGO
            # Apply Local Search Steepest Descent with NS1
            print(f"Applying Local Search Steepest Descent with NS1...")
            example_improved = LS_NS1_Steepest(example, w_cab10, c_cab10, alpha_val)
            print(f"Improved Solution: {example_improved[0]}")
            print(f"Improved Cost: {example_improved[1]:.4f}")
            print(f"Cost Improvement: {initial_cost - example_improved[1]:.4f}")
            print(f"Improvement %: {((initial_cost - example_improved[1])/initial_cost * 100):.2f}%")
    
    
    # ==============================================================================
    print("\n" + "="*60)
    print(" "*22 + "Dataset = CAB20")
    print("="*60)
    
    w_cab20 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,              
        usecols = "B:U",           
        skiprows = 32,             
        nrows = 20                 
    )
    w_cab20 = normalize_flow(w_cab20)
    
    c_cab20 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,             
        usecols = "B:U",          
        skiprows = 56,             
        nrows = 20                  
    )
    
    # Loop through each combination of p and alpha
    for p_val in p:
        for alpha_val in alpha:
            
            print("\n" + "="*60)
            print(f"Running with p={p_val} hubs and alpha={alpha_val}")
            print("="*60)
            
            # Generate initial solution
            example = initial_solution(20, p_val)
            print(f"\nInitial Solution: {example}")
            initial_cost = cost_evaluation(example, w_cab20, c_cab20, alpha_val)
            print(f"Initial Cost: {initial_cost:.4f}\n")
            
            # COPY THE BELOW PART FOR ADDING ANOTHER ALGO
            # Apply Local Search Steepest Descent with NS1
            print(f"Applying Local Search Steepest Descent with NS1...")
            example_improved = LS_NS1_Steepest(example, w_cab20, c_cab20, alpha_val)
            print(f"Improved Solution: {example_improved[0]}")
            print(f"Improved Cost: {example_improved[1]:.4f}")
            print(f"Cost Improvement: {initial_cost - example_improved[1]:.4f}")
            print(f"Improvement %: {((initial_cost - example_improved[1])/initial_cost * 100):.2f}%")
    
    
    # ==============================================================================
    print("\n" + "="*60)
    print(" "*22 + "Dataset = CAB25")
    print("="*60)
    
    w_cab25 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,              
        usecols = "B:Z",           
        skiprows = 80,             
        nrows = 25                 
    )
    w_cab25 = normalize_flow(w_cab25)
    
    c_cab25 = pd.read_excel(
        file_path,
        sheet_name = "CAB 10, 20 and 25Nodes",
        header = None,             
        usecols = "B:Z",          
        skiprows = 108,             
        nrows = 25                  
    )
    
    # Loop through each combination of p and alpha
    for p_val in p:
        for alpha_val in alpha:
            
            print("\n" + "="*60)
            print(f"Running with p={p_val} hubs and alpha={alpha_val}")
            print("="*60)
            
            # Generate initial solution
            example = initial_solution(25, p_val)
            print(f"\nInitial Solution: {example}")
            initial_cost = cost_evaluation(example, w_cab25, c_cab25, alpha_val)
            print(f"Initial Cost: {initial_cost:.4f}\n")
            
            # COPY THE BELOW PART FOR ADDING ANOTHER ALGO
            # Apply Local Search Steepest Descent with NS1
            print(f"Applying Local Search Steepest Descent with NS1...")
            example_improved = LS_NS1_Steepest(example, w_cab25, c_cab25, alpha_val)
            print(f"Improved Solution: {example_improved[0]}")
            print(f"Improved Cost: {example_improved[1]:.4f}")
            print(f"Cost Improvement: {initial_cost - example_improved[1]:.4f}")
            print(f"Improvement %: {((initial_cost - example_improved[1])/initial_cost * 100):.2f}%")
    
    
    
    plt.show()
    
    

if __name__ == "__main__":
    main()  # Execute the main function when the script is run