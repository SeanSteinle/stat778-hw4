#imports
import pandas as pd
import numpy as np
import os
import sys

from utils.transformations import logit, sigmoid
from utils.metrics import rmspe, tau
from utils.parallelize import all_models_repK, run_jobs

#parse arguments
if len(sys.argv) != 4: raise ValueError("invalid number of arguments supplied. either run like: python model_analysis.py or python model_analysis.py 5 10")
n_repeats, n_splits, jobdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] #TODO: make this clearer

#loading data
graduation_data_path = "data/inclass_activity_02-parallel-data.csv"
df = pd.read_csv(graduation_data_path)

#get number of cpus
n_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
n_cpus = int(n_cpus) if n_cpus != None else 1

#get job id, set output file, set specific seed for task
jobid = os.environ.get('SLURM_JOB_ID')
jobid = jobid if jobid != None else "unnamed_job"
np.random.seed(int(jobid)) #TODO: is this sufficient? i think so: https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
output_file = f"{jobdir}{jobid}.csv"
f = open(output_file, 'w')

#hm. looks like task/proc is always 0. creates 10 different job ids, starting at the first job id.
#so what we really need is a way to put all of these job ids into the same directory. needs to be done at the
#batch script level!

#WE DONT NEED PERFECTLY DEFAULT BEHAVIOR BETWEEN SING AND ARRAY!

#numerify categorical vars, transform target
categorical_features = ['school_type','magnet','urban_centric_locale','lunch_program']
for cf in categorical_features:
    if cf not in df.columns: continue #in case you're running again
    df = pd.get_dummies(df,prefix=[cf], columns=[cf])
df["logit_grad_rate"] = df["grad_rate"].apply(lambda p: logit(p))

feature_sets = [
    ['teachers_per_student',
     'support_per_teacher',
     'log_math_pass',
     'log_read_pass',
     'avg_salaries_admin_per_student',
     'avg_teacher_salaries',
     'school_type_Other/alternative',
     'school_type_Regular',
     'school_type_Vocational',
     'magnet_no',
     'magnet_yes',
     'urban_centric_locale_City',
     'urban_centric_locale_Rural',
     'urban_centric_locale_Suburb',
     'urban_centric_locale_Town',
     'lunch_program_no',
     'lunch_program_yes'],
    ['teachers_per_student',
     'support_per_teacher',
     'log_math_pass',
     'log_read_pass',
     'avg_salaries_admin_per_student',
     'school_type_Other/alternative',
     'school_type_Regular',
     'school_type_Vocational',
     'urban_centric_locale_City',
     'urban_centric_locale_Rural',
     'urban_centric_locale_Suburb',
     'urban_centric_locale_Town'],
    ['log_math_pass',
     'log_read_pass']
]

#run jobs!
partition_results = run_jobs(n_cpus, df, feature_sets, n_repeats, n_splits)

#aggregate results by model type
model_results = {i:{"rmspe":[],"tau":[]} for i in range(len(partition_results[0]))}
for partition_n,partition_data in enumerate(partition_results):
    for model_n,model_data in enumerate(partition_data): #model_n is 0 indexed
        model_results[model_n]["rmspe"].extend(model_data["test_rmspe"])
        model_results[model_n]["tau"].extend(model_data["test_tau"])
        
#display model results
f.write("model_index,rmspe,tau\n")
for model_n in range(len(model_results)): #TODO: the dimensionality here is unclear and its affecting my results...
    assert len(model_results[model_n]['rmspe']) == len(model_results[model_n]['tau'])
    for sample_n in range(len(model_results[model_n]['rmspe'])):
        f.write(f"{model_n},{model_results[model_n]['rmspe'][sample_n]},{model_results[model_n]['tau'][sample_n]}\n")
f.close()