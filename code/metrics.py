import numpy as np

def coverage(y_true, y_pred):
    return np.mean(y_true <= y_pred)

def requirement(y_true, y_pred):
    return np.mean(y_pred)

def exceeding(y_true, y_pred):
    return np.mean((y_true - y_pred)[y_true > y_pred])

def closeness(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))