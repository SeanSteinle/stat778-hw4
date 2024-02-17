#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00

module load hosts/hopper gnu10/10.3.0-ya
module load python/3.10.1-qb
mkdir "outputs/singular_runs/" -p
python model_analysis.py 50 10 "outputs/singular_runs/"
