from variable import Variable
from functions import square, exp 
import numpy as np

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

x = Variable(np.array(1.0))
x = Variable(None)
# x = Variable(1.0) ## 에러 발생