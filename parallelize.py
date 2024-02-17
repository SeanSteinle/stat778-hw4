import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from metrics import rmspe, tau

def all_models_repK(df: pd.DataFrame, feature_sets: list, n_repeats: int, n_splits: int, c: float=2.0):
    """Runs n_repeats repetitions of k-fold cross validations where k=n_splits."""
    scores = []
    metrics = {
        "rmspe": make_scorer(rmspe),
        "tau": make_scorer(tau)
    }
    for feature_set in feature_sets:
        X,y = df[feature_set],df["logit_grad_rate"]
        model = LinearRegression()
        rkf = RepeatedKFold(n_repeats=n_repeats,n_splits=n_splits)
        score = cross_validate(model, X, y, cv=rkf, scoring=metrics)
        scores.append(score)
    return scores

#define high-level parallelization
def make_groups(num_reps: int, num_jobs: int):
    """Divides num_reps by num_jobs such that no group is more than one larger than another."""
    r = num_reps%num_jobs
    groupsize = int(num_reps/num_jobs)
    groups = []
    for i in range(num_jobs):
        if r > 0:
            groups.append(groupsize+1)
            r-=1
        else:
            groups.append(groupsize)
    return groups

def run_jobs(n_jobs: int, df: pd.DataFrame, feature_sets: list, n_repeats: int=50, n_splits: int=10, c: float=2.0):
    """High-level function which parallelizes all_models_repK for n_jobs."""
    reps = make_groups(n_repeats, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(all_models_repK)(df, feature_sets, r, n_splits, c) for r in reps)
    return results

    #links for joblib
    #https://stackoverflow.com/questions/67237020/how-to-parallelize-multiple-model-building-procedures-in-sklearn
    #https://stackoverflow.com/questions/42220458/what-does-the-delayed-function-do-when-used-with-joblib-in-python