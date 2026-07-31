# linear_interpretable_surrogates

Code and data generator for the paper "Sparse Linear Surrogates for Interpretable Budget Allocation" by Marc Goerigk, Michael Hartisch, and Sebastian Merten.

**surrogates.py**: Contains the core implementation and the data generator.

**run_experiments.py**: A wrapper script used to execute specific experiments. This script is designed to be called from the command line, allowing parallel execution. 

**finding_alpha.py**: Contains the code used to determine suitable values for the regularization parameter ("alpha") of Lasso. 

**Requirements:**
- Python 3.12.3
- Gurobi 13.0.2 (License required)
- Dependencies: See **requirements.txt**

To run a specific experiment, call run_experiments.py with the required arguments:

```
python run_experiments.py [seed] [K] [time_limit] [N] [FS] [no_items] [FI] [binary_global] [binary_column] [binary_lasso]
```

The experiments in the paper can be reproduced by using the seeds 0, 1, ..., 24.