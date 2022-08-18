import numpy as np
from variable import Variable
from utils import as_array

class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.output = output
        return output
    
    def forward(self, x: np.ndarray):
        raise NotImplementedError()
    
    def backward(self, gy: np.ndarray):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)