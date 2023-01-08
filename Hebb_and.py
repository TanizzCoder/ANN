import numpy as np

INPUTS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
and_outputs = np.array([1, -1, -1, -1])
weights = np.array([0.0, 0.0])
bias = 0


def calculate_output(weights, instance, bias):
    sum = instance.dot(weights) + bias
    return step_function(sum)

def step_function(sum):
    if sum >= 0:
        return 1
    return -1


def hebb(outputs, weights, bias):
    for i in range(4):

        weights[0] = weights[0] + (INPUTS[i][0] * outputs[i])
        weights[1] = weights[1] + (INPUTS[i][1] * outputs[i])
        bias = bias + (1 * outputs[i])

        print("Weight updated: " + str(weights[0]))
        print("Weight updated: " + str(weights[1]))
        print("Bias updated: " + str(bias))
        print("----------------------------------------")

    return weights, bias



returned_weights, returned_bias = hebb(and_outputs, weights, bias)

print('prediction for [1, 1]: ' + str(calculate_output(returned_weights, np.array([[1, 1]]), returned_bias)))
print('prediction for [1, -1]: ' + str(calculate_output(returned_weights, np.array([[1, -1]]), returned_bias)))
print('prediction for [-1, 1]: ' + str(calculate_output(returned_weights, np.array([[-1, 1]]), returned_bias)))
print('prediction for [-1, -1]: ' + str(calculate_output(returned_weights, np.array([[-1, -1]]), returned_bias)))
