from math import sqrt
from statistics import median

def rmspe(preds, gt):
    """Finds the root mean-square prediction error between a list/series of predictions, preds, and a list/series of ground-truths, gt."""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist()
    sse = sum([(gt[i] - preds[i])**2 for i in range(len(gt))])
    return sqrt((1/len(gt))*sse)

def tau(preds, gt, c: float=2.0):
    """Finds the tau size of the prediction error given a list/series of predictions, preds, and a list/series of ground-truths, gt, and a cutoff c."""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    if c <= 0: return ValueError("the cutoff c must be greater than zero!")
    preds,gt = preds.tolist(),gt.tolist()
    median_error = median([abs(gt[i] - preds[i]) for i in range(len(gt))])
    sum_winsorized_error = sum([min(c,abs(gt[i] - preds[i])/median_error)**2 for i in range(len(gt))])
    return sqrt((1/len(gt))*sum_winsorized_error)