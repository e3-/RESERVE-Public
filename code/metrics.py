import numpy as np

def coverage(y_true, y_pred):
    return np.mean(y_true <= y_pred)

def requirement(y_true, y_pred):
    return np.mean(y_pred)

def exceeding(y_true, y_pred):
    return np.mean((y_true - y_pred)[y_true > y_pred])

def closeness(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def max_exceeding(y_true, y_pred):
    return np.max(y_true - y_pred)

def reserve_ramp_rate(y_true, y_pred):
    dt = 0.25 # hours (interval between time periods)
    return np.mean([np.abs(y_pred[i] - y_pred[i-1]) for i in range(1,len(y_pred))])/dt

def n_crossings(y_pred1, y_pred2):
    # y_pred1 is lower quantile estimate; y_pred2 is upper quantile estimate (crossing when y_pred1 > y_pred2)
    return np.sum(y_pred1 > y_pred2)