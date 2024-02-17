#imports
import pandas as pd
import numpy as np

from utils.transformations import logit, sigmoid
from utils.metrics import rmspe, tau
from utils.parallelize import all_models_repK, run_jobs

#loading data, # of cpus
graduation_data_path = "data/inclass_activity_02-parallel-data.csv"
df = pd.read_csv(graduation_data_path)
n_cpus = os.environ.get('SLURM_CPUS_PER_TASK')

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
print(f"running jobs!")
partition_results = run_jobs(n_cpus, df, feature_sets)
print(f"finished running jobs!")

#aggregate results by model type
model_results = {i:{"rmspe":[],"tau":[]} for i in range(len(partition_results[0]))}
for partition_n,partition_data in enumerate(partition_results):
    for model_n,model_data in enumerate(partition_data): #model_n is 0 indexed
        model_results[model_n]["rmspe"].extend(model_data["test_rmspe"])
        model_results[model_n]["tau"].extend(model_data["test_tau"])
        
#display model results
for i in range(len(model_results)):
    print(f"Model #{i+1} results:\nAverage RMSPE:{np.mean(model_results[i]['rmspe'])}\tavg tau: {np.mean(model_results[i]['tau'])}")

#TODO: write results to file?