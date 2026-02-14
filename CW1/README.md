# Project Title

A brief description of what this project does and its purpose.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Sarthak-Vishnu/B286266_code.git
   ```
2. Navigate to the project directory:
   ```
   cd CW1
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
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


## Testing

To run the tests, navigate to the `B286266_code\CW1` directory and execute:
```
pytest tests/test_algorithms.py -v
```

<!-- ## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
