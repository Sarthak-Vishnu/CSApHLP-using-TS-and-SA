"""
Script to check for NaN values in the cost matrices
"""

import pandas as pd
import os

def load_dataset(dataset_name='CAB10'):
    """
    Load dataset and check for NaN values
    
    Args:
        dataset_name (str): Name of dataset ('CAB10', 'CAB20', or 'CAB25')
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
        w_skiprows = 5
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
    
    # Load cost matrix
    c = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        skiprows=c_skiprows,
        nrows=nrows
    )
    
    print(f"\n{'='*70}")
    print(f"Checking {dataset_name} Cost Matrix")
    print(f"{'='*70}")
    print(f"\nOriginal data type: {c.dtypes.unique()}")
    print(f"\nFirst few rows:")
    print(c.head())
    
    # Check for NaN before conversion
    nan_before = c.isna().sum().sum()
    print(f"\nNaN values BEFORE conversion: {nan_before}")
    
    # Convert to numeric
    c = c.apply(pd.to_numeric, errors='coerce')
    
    nan_after = c.isna().sum().sum()
    print(f"NaN values AFTER conversion: {nan_after}")
    
    if nan_after > 0:
        print(f"\nNaN values found in the following positions:")
        nan_positions = c.isna()
        for i in range(len(c)):
            for j in range(len(c.columns)):
                if nan_positions.iloc[i, j]:
                    print(f"  Position [{i}, {j}] (Node {i+1}, Node {j+1}): {c.iloc[i, j]}")
    else:
        print("\n✓ No NaN values found!")
    
    print(f"\nData type after conversion: {c.dtypes.unique()}")
    print(f"\nFirst few rows after conversion:")
    print(c.head())
    
    return c


if __name__ == "__main__":
    print("Checking for NaN values in cost matrices...\n")
    
    # Check all datasets
    # for dataset in ['CAB10', 'CAB20', 'CAB25']:
    for dataset in ['TR40', 'TR55', 'RGP100']:
        try:
            load_dataset(dataset)
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
