#
import numpy
import torch


#
"""
def prepare_inputs(X, Y, window):
    xx = []
    yy = []
    for j in range(X.shape[0] - window + 1):
        xx.append(X[j:j + window, :].reshape(1, window, X.shape[1]))
        yy.append(Y[j:j + window].reshape(1, window, 1))
    xx = numpy.concatenate(xx, axis=0)
    yy = numpy.concatenate(yy, axis=0)
    xx = torch.tensor(xx, dtype=torch.float)
    yy = torch.tensor(yy, dtype=torch.float)
    return xx, yy
"""

def prepare_inputs(X, Y, window):
    xx = []
    # yy = []
    for j in range(X.shape[0] - window):
        xx.append(X[j:j + window, :].reshape(1, window, X.shape[1]))
        # yy.append(Y[j:j + window].reshape(1, window, 1))
    xx = numpy.concatenate(xx, axis=0)
    yy = Y[window:]
    xx = torch.tensor(xx, dtype=torch.float)
    yy = torch.tensor(yy, dtype=torch.float)
    return xx, yy

def simulate_lags(X, Y, window):
    xx = []
    # yy = []
    for j in range(window):
        xx.append(X[j:X.shape[0]-window+j, :])
        # yy.append(Y[j:j + window].reshape(1, window, 1))
    xx = numpy.concatenate(xx, axis=1)
    yy = Y[window:]
    xx = torch.tensor(xx, dtype=torch.float)
    yy = torch.tensor(yy, dtype=torch.float)
    return xx, yy
