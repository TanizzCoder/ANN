import numpy as np
X_XOR = np.array([[0,0,1], [0,1,1], [1,0,1],[1,1,1]]) 
y_truth = np.array([[0],[1],[1],[0]])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_der(output):
    return output * (1 - output)
np.random.seed(1)
syn_0 = 2*np.random.random((3,4)) - 1
syn_1 = 2*np.random.random((4,1)) - 1
syn_1
for i in range(60000):
    layer_1 = sigmoid(X_XOR.dot(syn_0))
    layer_2 = sigmoid(layer_1.dot(syn_1))
    error = layer_2 - y_truth
    layer_2_delta = error * sigmoid_der(layer_2)
    layer_1_error = layer_2_delta.dot(syn_1.T)
    layer_1_delta = layer_1_error * sigmoid_der(layer_1)
    syn_1 -= layer_1.T.dot(layer_2_delta)
    syn_0 -= X_XOR.T.dot(layer_1_delta)
    if i % 10000 == 1:
        print(layer_2)


layer_2_delta
sigmoid_der(layer_1)
