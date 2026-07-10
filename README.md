# AQUA-PLATFORM-TECHNICAL-HIRING-CHALLENGE
## Testing
To execute the automated gradient checking test suite, run:
```bash
python -m tests.test_grad_check
To execute the masked-weight correctness verification test, run:
```bash
python -m tests.test_masked_weight
## Execution
To reproduce Part 2 baseline dense network model training, run:
```bash
python -m train.train_baseline
To execute the Part 3 Taylor-expansion self-pruning training track, run:
```bash
python -m train.train_pruning
## Pareto Experiments Execution
To reproduce the Part 4 Pareto sweep, variance assessment across seeds, and graph generation, run:
```bash
python -m train.run_experiments