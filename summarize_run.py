#this script should pull together the results of the run and summarize them!

#imports
import os
import sys
import pandas as pd

if len(sys.argv) != 2: raise ValueError("invalid number of arguments supplied. run like: python summarize_run outputs/singular_runs/")
results_dir = sys.argv[1]

#load all .csvs in directory into a single df
dfs = []
for results_file in os.listdir(results_dir):
    if results_file[:-3] != ".csv": continue #skip if not csv
    print(results_file)
    dfs.append(pd.read_csv(results_dir+results_file))
df = pd.concat(dfs)
