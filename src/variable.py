from pyclbr import Function
import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: Function):
        self.creator = func