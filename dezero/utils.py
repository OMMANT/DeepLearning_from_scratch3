import numpy as np
from dezero.variable import Variable
from dezero.functions import *

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x