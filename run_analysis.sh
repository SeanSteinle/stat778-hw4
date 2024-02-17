#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00

module load hosts/hopper gnu10/10.3.0-ya
module load python/3.10.1-qb

python model_analysis.py

#TODO: should this create a venv and install everything necessary?
#source ~/venvs/hw4/bin/activate

#Write a SLURM script to run your cross-validation script from Homework 3 on the Hopper cluster. 
#Your SLURM script should request 4 CPUs, and your analysis should use this many CPUs for parallel 
#computation of the repeated cross-validation. Use the value of the environment variable SLURM_CPUS_PER_TASK 
#to know the number of available CPUs in your analysis script.