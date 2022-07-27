import numpy as np
from variable import Variable

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)

        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return np.square(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)