
# importing Python library
import numpy as np
  
# define Unit Step Function
def unitStep(v):
    if v <= 0:
        return 0
    else:
        return 1
  
# design Perceptron Model
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y
  
# OR Logic Function
# w1 = 1, w2 = 1, b = -0.5
def AND_logicFunction(x):
    w = np.array([1, 1])
    b = -1
    return perceptronModel(x, w, b)
def XNOR_logicFunction(x):
    x1=AND_logicFunction(x)
    x2=NOR_logicFunction(x)
    x3=np.array([x1,x2])
    w=np.array([2,2])
    b=-1
    return perceptronModel(x3,w,b)
def NOR_logicFunction(x):
    w=np.array([-1,-1])
    b=1
    return perceptronModel(x,w,b)

# testing the Perceptron Model
test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])
print("XNOR({}, {}) = {}".format(0, 0, XNOR_logicFunction(test1)))
print("XNOR({}, {}) = {}".format(0, 1, XNOR_logicFunction(test2)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_logicFunction(test3)))
print("XNOR({}, {}) = {}".format(1, 1, XNOR_logicFunction(test4)))
