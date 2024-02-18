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
    if results_file[-4:] != ".csv": continue #skip if not csv
    dfs.append(pd.read_csv(results_dir+results_file))
if len(dfs) == 0: 
    print("no .csv files found in directory!")
    sys.exit(0)
elif len(dfs) == 1:
    df = dfs[0]
elif len(dfs) > 1:
    df = pd.concat(dfs)

print(df.head())
print(len(df)) #each row represents a single batch of 
print(df.columns) #we track 3 stats for every model