# Solving Capacitated Single Allocation p-Hub Location Problem using Metaheuristics Approaches

Abstract: </br>
This study investigates two metaheuristic approaches namely Tabu Search (TS) and Simulated Annealing (SA) applied to CSApHLP, a combinatorial optimisation problem arising in logistics and transportation network design. Both algorithms are evaluated across six benchmark datasets ranging from 10 to 100 nodes. Simulated Annealing employs a geometric cooling schedule with an adaptively computed initial temperature, while Tabu Search uses a tabu list with aspiration criteria to guide the search process. Results show that TS performs more consistently on smaller datasets, whereas SA achieves comparable or better solution quality on larger problems at up to $17\times$ lower computational time. Sensitivity analysis confirms that the inter-hub discount factor $\alpha$ has a predictable influence on total network cost across all instances.

## Installation

1. Navigate to the project directory:
   ```
   cd CW1
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command from the  `B286266_code\CW1` directory:
```
python .\src\main_run.py
```

To run the parameter tuning for Tabu Search or Simulated Annealing, navigate to the `B286266_code\CW1` directory and execute the corresponding commads:
```
python -m src.parameter_tuning_tabu
python -m src.parameter_tuning_SA
```

This generates a results file at:
```
results/[Tabu/SA]_all_results.csv
```

To check and visualize the cooling schedule curve used in Simulated Annealing, run:
```
python -m src.debug_sa
```


## Testing

To run the tests, navigate to the `B286266_code\CW1` directory and execute:
```
pytest tests/test_algorithms.py -v
```

<!-- ## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
