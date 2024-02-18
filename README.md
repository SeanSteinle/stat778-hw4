# STAT778 Homework 4
*Sean Steinle*

This is my submission for HW4 for STAT778 at GMU.

## How To Run
0. You should have no issues with dependencies if you're running on GMU's HPC cluster, Hopper. I've only used basic Python data science libraries that are included in the Python version I've loaded.
1. If you'd like to produce fresh results, empty the `outputs/` directory and run one or both of `sbatch run_analysis_singular.sh` or `sbatch run_analysis_array.sh`. The script will run a parallelized computation of repeated k-fold cross-validation, dumping the results into `outputs/singular_runs/` and `outputs/array_runs`. If you'd like to analyze the sample data in this repository, you can skip this step!
    - To change the number of replications, folds, or array size (for array analysis only), modify the arguments to the `model_analysis.py` script that is invoked in the `run_analysis*` bash scripts.
    - If you use `sbatch` to run these scripts, you'll get a bunch of `slurm-<taskid>.out` files that litter your directory. You can delete these easily with `rm slurm-<startoftaskid>*`.
2. After you have data to analyze in the `outputs/` directory, you can run the `summarize_run.ipynb` notebook to load the data into a dataframe and do a simple comparison of the data.

## Other Notes
- If it isn't obvious, `run_analysis_singular.sh` completes problem #1 and `run_analysis_array.sh` completes problem #2!
- Helper functions for the parallelized computation can be foudn in `utils/`. Additionally, if you'd like to implement more evaluation metrics, parallelization utilities, or transformation functions, put them here.