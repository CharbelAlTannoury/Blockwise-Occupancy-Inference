# Blockwise occupancy inference under weak dependence

Technical appendix and simulation code for **Blockwise Occupancy Inference under Weak Dependence with Record-Based Persistence Thresholds**, by Charbel Al Tannoury, Anis Hoayek, Youssef Salman, Hyam Abboud, and Mireille Batton-Hubert.

## Contents

- [Technical appendix](supplement.pdf)
- `simulation_study.py`: fitted CHARN experiments, exact count comparisons, and diagnostic-corruption experiments.
- `calibration_sensitivity.py`: reference-length and gap experiments, fitted rank comparisons, and numerical mixing bounds.
- `count_distributions.py`: distribution calculations used by the simulation programs.
- `requirements.txt`: Python package versions.

## Reproducing the experiments

The experiments use Python 3.12. Keep the three Python files in the same directory, then run:

```bash
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 simulation_study.py --replications 1000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 calibration_sensitivity.py --replications 1000
```

The programs use fixed seeds: 20260919 for the main simulation study and 20261107 for the calibration-sensitivity study. They generate the tables and figures under `tables/` and `figures/`, together with `experiment_metadata.json` and `sensitivity_metadata.json`. No external dataset is required.

For a shorter run with separate output files:

```bash
python3 calibration_sensitivity.py --replications 80 --output pilot-results
```

The fitted rank comparisons use a common nominal level and report the achieved empirical false-alarm rates. Small floating-point differences may occur across platforms.
