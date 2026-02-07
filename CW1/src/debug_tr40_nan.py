"""
Debug script for TR40 NaN cost issues.
Loads TR40 using the same parameters as parameter_tuning.py and reports NaNs.
"""

import os
import pandas as pd
from .functions import normalize_flow, cost_evaluation


def load_tr40():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    file_path = os.path.join(data_dir, "CAB_TR_and RGP Datasets_2026.xlsx")

    n = 40
    w = pd.read_excel(
        file_path,
        sheet_name="TR 40 and 55 Nodes",
        header=None,
        usecols="C:AP",
        skiprows=4,
        nrows=40,
    )
    w = normalize_flow(w)

    c = pd.read_excel(
        file_path,
        sheet_name="TR 40 and 55 Nodes",
        header=None,
        usecols="C:AP",
        skiprows=47,
        nrows=40,
    )
    c = c.apply(pd.to_numeric, errors="coerce")

    return w, c, n


def report_nan_locations(df, name):
    nan_count = df.isna().sum().sum()
    print(f"{name} NaN count: {nan_count}")
    if nan_count:
        nan_positions = df.isna()
        rows, cols = nan_positions.to_numpy().nonzero()
        # Print first 20 NaN positions for brevity
        print(f"{name} first NaN positions (up to 20):")
        for i, j in list(zip(rows, cols))[:20]:
            print(f"  ({i}, {j}) -> {df.iloc[i, j]}")


def main():
    w, c, n = load_tr40()
    print("TR40 load complete")
    print(f"w shape: {w.shape}, c shape: {c.shape}")

    report_nan_locations(w, "w")
    report_nan_locations(c, "c")

    # Quick cost check with a simple feasible solution (hub assignment to itself)
    solution = list(range(1, n + 1))
    cost = cost_evaluation(solution, w, c, 0.3)
    print(f"Cost check (identity hubs): {cost}")


if __name__ == "__main__":
    main()
