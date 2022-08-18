from pyclbr import Function
import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        
        while funcs:
            f = funcs.pop() # 함수를 가져온다
            x, y = f.input, f.output    # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad) # backward 메서드를 호출한다.

            if x.creator is not None:
                funcs.append(x.creator)