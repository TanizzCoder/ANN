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
def OR_logicFunction(x):
    w = np.array([2, 2])
    b = -1
    return perceptronModel(x, w, b)
def XOR_logicFunction(x):
    x1=OR_logicFunction(x)
    x2=NAND_logicFunction(x)
    x3=np.array([x1,x2])
    w=np.array([1,1])
    b=-1
    return perceptronModel(x3,w,b)
def NAND_logicFunction(x):
    w=np.array([-1,-1])
    b=2
    return perceptronModel(x,w,b)
 
# testing the Perceptron Model
test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])
print("XOR({}, {}) = {}".format(0, 0, XOR_logicFunction(test1)))
print("XOR({}, {}) = {}".format(0, 1, XOR_logicFunction(test2)))
print("XOR({}, {}) = {}".format(1, 0, XOR_logicFunction(test3)))
print("XOR({}, {}) = {}".format(1, 1, XOR_logicFunction(test4)))
     
