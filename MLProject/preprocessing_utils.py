import numpy as np

def clip_age(X):
    return np.clip(X, a_min=None, a_max=65)

def clip_emplen(X):
    return np.clip(X, a_min=None, a_max=40)

def clip_loanpct(X):
    return np.clip(X, a_min=None, a_max=0.6)

def log1p_clip(X):
    X_log = np.log1p(X)
    clip_val = np.nanpercentile(X_log, 99)
    return np.clip(X_log, a_min=None, a_max=clip_val)